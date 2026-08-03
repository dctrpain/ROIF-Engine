from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.future_plane_analyzer import (
    CandidatePlaneAnalysis,
    FuturePlane,
    FuturePlaneAnalysisResult,
    FuturePlaneAnalysisStatus,
    FuturePlaneAnalyzer,
    FuturePlaneAnalyzerError,
    FuturePlaneContribution,
    FuturePlaneDirection,
    FuturePlaneScope,
    FuturePlaneStatus,
    TemporalProfile,
)
from roif.history.history_forecast import (
    ForecastCandidate,
    ForecastCandidateStatus,
    ForecastDirection,
    ForecastHorizon,
    PlaneInfluence,
    PlaneInfluenceSet,
)
from roif.history.structural_signature import (
    StructuralSignature,
)


def make_signature(
    signature_id: str = "signature-future",
    **overrides,
) -> StructuralSignature:
    data = {
        "signature_id": signature_id,
        "source_pattern_id": f"pattern-{signature_id}",
        "label": signature_id,
        "plane_profile": {
            "mechanical": 0.70,
            "thermal": 0.30,
        },
        "agent_profile": {
            "environment": 0.60,
            "internal": 0.40,
        },
        "kind_profile": {
            "damage": 0.60,
            "remodeling": 0.40,
        },
        "time_scale_profile": {
            "medium": 0.50,
            "slow": 0.50,
        },
        "target_profile": {
            "node:A": 0.60,
            "node:B": 0.40,
        },
        "total_changes": 3,
        "root_count": 1,
        "leaf_count": 1,
        "causal_depth": 3,
        "start_time": 0.0,
        "end_time": 30.0,
        "duration": 30.0,
        "total_capacity_loss": 0.25,
        "total_capacity_gain": 0.08,
        "net_capacity_effect": -0.17,
        "cumulative_signature_weight": 1.80,
        "mean_signature_weight": 0.60,
        "persistence_index": 0.70,
        "irreversibility_index": 0.55,
        "progression_index": 0.75,
        "adaptation_index": 0.25,
        "harmful_fraction": 0.65,
        "beneficial_fraction": 0.25,
        "mixed_fraction": 0.10,
        "metadata": {
            "source": "future-plane-test",
        },
    }

    data.update(overrides)
    return StructuralSignature(**data)


def make_candidate(
    candidate_id: str = "candidate-a",
    *,
    start: float = 0.0,
    end: float = 30.0,
    affected_target_ids: tuple[str, ...] = ("node:A",),
    affected_plane_ids: tuple[str, ...] = ("mechanical",),
    **overrides,
) -> ForecastCandidate:
    data = {
        "candidate_id": candidate_id,
        "name": f"Candidate {candidate_id}",
        "horizon": ForecastHorizon(
            start=start,
            end=end,
            units="days",
        ),
        "predicted_signature": make_signature(
            signature_id=f"signature-{candidate_id}"
        ),
        "description": "Future structural candidate.",
        "direction": ForecastDirection.MIXED,
        "affected_plane_ids": affected_plane_ids,
        "affected_target_ids": affected_target_ids,
        "predicted_change_kinds": (
            "remodeling",
        ),
        "prior_probability": 0.60,
        "validation_score": 0.90,
        "complexity": 1.0,
        "custom_penalty": 0.0,
        "status": (
            ForecastCandidateStatus.FORWARD_VALIDATED
        ),
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return ForecastCandidate(**data)


def make_plane(
    future_plane_id: str = "future-plane-a",
    *,
    plane_id: str = "mechanical",
    start: float = 0.0,
    end: float = 30.0,
    signed_effect: float = 0.50,
    target_ids: tuple[str, ...] = ("node:A",),
    candidate_ids: tuple[str, ...] = (),
    incompatible_candidate_ids: tuple[str, ...] = (),
    **overrides,
) -> FuturePlane:
    data = {
        "future_plane_id": future_plane_id,
        "plane_id": plane_id,
        "name": f"Future plane {future_plane_id}",
        "horizon": ForecastHorizon(
            start=start,
            end=end,
            units="days",
        ),
        "signed_effect": signed_effect,
        "intensity": 1.0,
        "confidence": 0.90,
        "multiplier_scale": 1.0,
        "direction": FuturePlaneDirection.AMPLIFY,
        "temporal_profile": TemporalProfile.CONSTANT,
        "scope": FuturePlaneScope.EXTERNAL,
        "target_ids": target_ids,
        "candidate_ids": candidate_ids,
        "incompatible_candidate_ids": (
            incompatible_candidate_ids
        ),
        "operational_risk": 0.10,
        "cascade_risk": 0.10,
        "observability": 0.90,
        "reversibility": 0.90,
        "non_fonit_allowed": True,
        "status": FuturePlaneStatus.ACTIVE,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return FuturePlane(**data)


def make_contribution(
    *,
    future_plane_id: str = "future-plane-a",
    applied: bool = True,
    veto_reason: str | None = None,
) -> FuturePlaneContribution:
    return FuturePlaneContribution(
        future_plane=make_plane(
            future_plane_id=future_plane_id,
        ),
        candidate_id="candidate-a",
        temporal_weight=1.0 if applied else 0.0,
        target_weight=1.0 if applied else 0.0,
        confidence_weight=0.90,
        observability_weight=0.90,
        safety_weight=0.90,
        effective_exponent=0.30 if applied else 0.0,
        effective_multiplier=1.3498588076 if applied else 1.0,
        applied=applied,
        veto_reason=veto_reason,
    )

# ---------------------------------------------------------------------------
# FuturePlane
# ---------------------------------------------------------------------------


def test_future_plane_creation() -> None:
    plane = make_plane()

    assert plane.future_plane_id == "future-plane-a"
    assert plane.plane_id == "mechanical"
    assert plane.direction is FuturePlaneDirection.AMPLIFY
    assert plane.temporal_profile is TemporalProfile.CONSTANT
    assert plane.scope is FuturePlaneScope.EXTERNAL
    assert plane.status is FuturePlaneStatus.ACTIVE
    assert plane.is_eligible is True


def test_future_plane_accepts_string_enums() -> None:
    plane = make_plane(
        direction="attenuate",
        temporal_profile="ramp_up",
        scope="target",
        status="inferred",
    )

    assert plane.direction is FuturePlaneDirection.ATTENUATE
    assert plane.temporal_profile is TemporalProfile.RAMP_UP
    assert plane.scope is FuturePlaneScope.TARGET
    assert plane.status is FuturePlaneStatus.INFERRED


def test_future_plane_is_immutable() -> None:
    plane = make_plane()

    with pytest.raises(FrozenInstanceError):
        plane.intensity = 2.0


def test_future_plane_metadata_is_read_only() -> None:
    plane = make_plane()

    with pytest.raises(TypeError):
        plane.metadata["new"] = "value"


def test_future_plane_ids_are_deduplicated() -> None:
    plane = make_plane(
        target_ids=(
            "node:A",
            "node:B",
            "node:A",
        ),
        candidate_ids=(
            "candidate-a",
            "candidate-b",
            "candidate-a",
        ),
        incompatible_candidate_ids=(
            "candidate-x",
            "candidate-y",
            "candidate-x",
        ),
    )

    assert plane.target_ids == (
        "node:A",
        "node:B",
    )
    assert plane.candidate_ids == (
        "candidate-a",
        "candidate-b",
    )
    assert plane.incompatible_candidate_ids == (
        "candidate-x",
        "candidate-y",
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (FuturePlaneStatus.ACTIVE, True),
        (FuturePlaneStatus.INFERRED, True),
        (FuturePlaneStatus.INCOMPLETE, True),
        (FuturePlaneStatus.DISABLED, False),
        (FuturePlaneStatus.FAILED, False),
        (FuturePlaneStatus.VETOED, False),
    ),
)
def test_future_plane_eligibility(
    status: FuturePlaneStatus,
    expected: bool,
) -> None:
    plane = make_plane(
        status=status,
    )

    assert plane.is_eligible is expected


def test_non_fonit_gate_makes_plane_ineligible() -> None:
    plane = make_plane(
        non_fonit_allowed=False,
    )

    assert plane.is_eligible is False


def test_future_plane_safety_penalty() -> None:
    plane = make_plane(
        operational_risk=0.30,
        cascade_risk=0.60,
        reversibility=0.50,
    )

    assert plane.safety_penalty == pytest.approx(
        (0.30 + 0.60 + 0.50) / 3.0
    )


def test_positive_effect_produces_multiplier_above_one() -> None:
    plane = make_plane(
        signed_effect=0.50,
        intensity=2.0,
        multiplier_scale=1.0,
    )

    assert plane.base_multiplier > 1.0


def test_negative_effect_produces_multiplier_below_one() -> None:
    plane = make_plane(
        signed_effect=-0.50,
        intensity=2.0,
        multiplier_scale=1.0,
        direction=FuturePlaneDirection.ATTENUATE,
    )

    assert 0.0 < plane.base_multiplier < 1.0


def test_neutral_effect_produces_multiplier_one() -> None:
    plane = make_plane(
        signed_effect=0.0,
        direction=FuturePlaneDirection.NEUTRAL,
    )

    assert plane.base_multiplier == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_value",
    (
        -1.01,
        1.01,
        float("inf"),
        float("nan"),
    ),
)
def test_future_plane_rejects_invalid_signed_effect(
    bad_value: float,
) -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="signed_effect",
    ):
        make_plane(
            signed_effect=bad_value,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "confidence",
        "operational_risk",
        "cascade_risk",
        "observability",
        "reversibility",
    ),
)
def test_future_plane_rejects_invalid_unit_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match=field_name,
    ):
        make_plane(
            **{
                field_name: 1.01,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "intensity",
    ),
)
def test_future_plane_rejects_negative_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match=field_name,
    ):
        make_plane(
            **{
                field_name: -0.01,
            }
        )


def test_future_plane_rejects_nonpositive_multiplier_scale() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="multiplier_scale",
    ):
        make_plane(
            multiplier_scale=0.0,
        )


def test_future_plane_rejects_nonboolean_gate() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="non_fonit_allowed must be a boolean",
    ):
        make_plane(
            non_fonit_allowed=1,
        )


def test_future_plane_rejects_invalid_enum() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="unsupported future-plane enum value",
    ):
        make_plane(
            temporal_profile="unsupported",
        )


def test_future_plane_round_trip() -> None:
    plane = make_plane()

    restored = FuturePlane.from_dict(
        plane.to_dict()
    )

    assert restored == plane
    assert restored is not plane


def test_future_plane_from_dict_rejects_version() -> None:
    data = make_plane().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="unsupported future plane version",
    ):
        FuturePlane.from_dict(data)


# ---------------------------------------------------------------------------
# Temporal behavior and applicability
# ---------------------------------------------------------------------------


def test_constant_temporal_weight_full_overlap() -> None:
    plane = make_plane(
        start=0.0,
        end=30.0,
        temporal_profile=TemporalProfile.CONSTANT,
    )
    candidate = make_candidate(
        start=0.0,
        end=30.0,
    )

    assert plane.temporal_weight(
        candidate.horizon
    ) == pytest.approx(1.0)


def test_constant_temporal_weight_partial_overlap() -> None:
    plane = make_plane(
        start=0.0,
        end=15.0,
    )
    candidate = make_candidate(
        start=0.0,
        end=30.0,
    )

    assert plane.temporal_weight(
        candidate.horizon
    ) == pytest.approx(0.5)


def test_temporal_weight_no_overlap() -> None:
    plane = make_plane(
        start=40.0,
        end=50.0,
    )
    candidate = make_candidate(
        start=0.0,
        end=30.0,
    )

    assert plane.temporal_weight(
        candidate.horizon
    ) == pytest.approx(0.0)


def test_ramp_up_and_ramp_down_weights_differ() -> None:
    candidate = make_candidate(
        start=0.0,
        end=10.0,
    )
    ramp_up = make_plane(
        start=0.0,
        end=20.0,
        temporal_profile=TemporalProfile.RAMP_UP,
    )
    ramp_down = make_plane(
        future_plane_id="future-plane-down",
        start=0.0,
        end=20.0,
        temporal_profile=TemporalProfile.RAMP_DOWN,
    )

    assert (
        ramp_down.temporal_weight(candidate.horizon)
        >
        ramp_up.temporal_weight(candidate.horizon)
    )


def test_impulse_weight_is_bounded() -> None:
    plane = make_plane(
        temporal_profile=TemporalProfile.IMPULSE,
    )
    candidate = make_candidate()

    value = plane.temporal_weight(
        candidate.horizon
    )

    assert 0.0 <= value <= 1.0


def test_plane_applies_to_matching_candidate() -> None:
    plane = make_plane(
        candidate_ids=("candidate-a",),
        target_ids=("node:A",),
    )
    candidate = make_candidate(
        candidate_id="candidate-a",
        affected_target_ids=("node:A",),
    )

    assert plane.applies_to(candidate) is True


def test_plane_rejects_incompatible_candidate() -> None:
    plane = make_plane(
        incompatible_candidate_ids=(
            "candidate-a",
        ),
    )

    assert plane.applies_to(
        make_candidate()
    ) is False


def test_plane_rejects_nonlisted_candidate() -> None:
    plane = make_plane(
        candidate_ids=("candidate-b",),
    )

    assert plane.applies_to(
        make_candidate(
            candidate_id="candidate-a",
        )
    ) is False


def test_plane_rejects_nonmatching_target() -> None:
    plane = make_plane(
        target_ids=("node:Z",),
    )
    candidate = make_candidate(
        affected_target_ids=("node:A",),
    )

    assert plane.applies_to(candidate) is False


def test_plane_rejects_nonoverlapping_horizon() -> None:
    plane = make_plane(
        start=40.0,
        end=50.0,
    )

    assert plane.applies_to(
        make_candidate(
            start=0.0,
            end=30.0,
        )
    ) is False


# ---------------------------------------------------------------------------
# Contribution
# ---------------------------------------------------------------------------


def test_contribution_creation() -> None:
    contribution = make_contribution()

    assert contribution.applied is True
    assert contribution.effective_multiplier > 1.0
    assert contribution.veto_reason is None


def test_contribution_is_immutable() -> None:
    contribution = make_contribution()

    with pytest.raises(FrozenInstanceError):
        contribution.applied = False


def test_contribution_to_plane_influence() -> None:
    contribution = make_contribution()

    influence = contribution.to_plane_influence()

    assert isinstance(
        influence,
        PlaneInfluence,
    )
    assert influence.plane_id == "mechanical"
    assert influence.multiplier == pytest.approx(
        contribution.effective_multiplier
    )
    assert influence.metadata["candidate_id"] == (
        "candidate-a"
    )


def test_contribution_rejects_invalid_future_plane() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="future_plane must be a FuturePlane",
    ):
        FuturePlaneContribution(
            future_plane="plane",
            candidate_id="candidate-a",
            temporal_weight=1.0,
            target_weight=1.0,
            confidence_weight=1.0,
            observability_weight=1.0,
            safety_weight=1.0,
            effective_exponent=0.0,
            effective_multiplier=1.0,
            applied=True,
        )


def test_contribution_rejects_invalid_multiplier() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="effective_multiplier",
    ):
        FuturePlaneContribution(
            future_plane=make_plane(),
            candidate_id="candidate-a",
            temporal_weight=1.0,
            target_weight=1.0,
            confidence_weight=1.0,
            observability_weight=1.0,
            safety_weight=1.0,
            effective_exponent=0.0,
            effective_multiplier=0.0,
            applied=True,
        )


# ---------------------------------------------------------------------------
# CandidatePlaneAnalysis and result
# ---------------------------------------------------------------------------


def test_candidate_analysis_creation() -> None:
    contribution = make_contribution()

    analysis = CandidatePlaneAnalysis(
        analysis_id="analysis-a",
        candidate_id="candidate-a",
        contributions=(contribution,),
        combined_multiplier=(
            contribution.effective_multiplier
        ),
        combined_log_multiplier=0.30,
        applied_plane_count=1,
        vetoed_plane_count=0,
        mean_observability=0.90,
        mean_safety=0.90,
    )

    assert analysis.analysis_id == "analysis-a"
    assert analysis.applied_plane_count == 1
    assert analysis.combined_multiplier > 1.0


def test_candidate_analysis_is_immutable() -> None:
    analysis = CandidatePlaneAnalysis(
        candidate_id="candidate-a",
    )

    with pytest.raises(FrozenInstanceError):
        analysis.applied_plane_count = 2


def test_candidate_analysis_rejects_duplicate_planes() -> None:
    contribution = make_contribution()

    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="future_plane_id values must be unique",
    ):
        CandidatePlaneAnalysis(
            candidate_id="candidate-a",
            contributions=(
                contribution,
                contribution,
            ),
        )


def test_candidate_analysis_builds_influence_set() -> None:
    applied = make_contribution(
        future_plane_id="future-plane-a",
    )

    vetoed = make_contribution(
        future_plane_id="future-plane-b",
        applied=False,
        veto_reason="non_fonit_gate",
    )
    analysis = CandidatePlaneAnalysis(
        candidate_id="candidate-a",
        contributions=(
            applied,
            vetoed,
        ),
        combined_multiplier=(
            applied.effective_multiplier
        ),
        combined_log_multiplier=0.30,
        applied_plane_count=1,
        vetoed_plane_count=1,
    )

    influence_set = (
        analysis.plane_influence_set
    )

    assert isinstance(
        influence_set,
        PlaneInfluenceSet,
    )
    assert len(influence_set.influences) == 1
    assert influence_set.influences[0].plane_id == (
        "mechanical"
    )


def test_analysis_result_creation() -> None:
    analysis = CandidatePlaneAnalysis(
        candidate_id="candidate-a",
    )

    result = FuturePlaneAnalysisResult(
        result_id="result-a",
        status=FuturePlaneAnalysisStatus.COMPLETED,
        candidate_analyses=(analysis,),
        plane_count=1,
        eligible_plane_count=1,
        vetoed_plane_count=0,
        metadata={
            "source": "test",
        },
    )

    assert result.result_id == "result-a"
    assert result.for_candidate(
        "candidate-a"
    ) == analysis


def test_analysis_result_metadata_is_read_only() -> None:
    result = FuturePlaneAnalysisResult(
        status=FuturePlaneAnalysisStatus.NO_PLANES,
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_analysis_result_rejects_duplicate_candidates() -> None:
    first = CandidatePlaneAnalysis(
        candidate_id="duplicate",
    )
    second = CandidatePlaneAnalysis(
        candidate_id="duplicate",
    )

    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="candidate_id values must be unique",
    ):
        FuturePlaneAnalysisResult(
            status=FuturePlaneAnalysisStatus.COMPLETED,
            candidate_analyses=(
                first,
                second,
            ),
        )


def test_analysis_result_to_dict() -> None:
    result = FuturePlaneAnalysisResult(
        result_id="result-a",
        status=FuturePlaneAnalysisStatus.NO_PLANES,
    )

    data = result.to_dict()

    assert data["version"] == "1.0.0"
    assert data["result_id"] == "result-a"
    assert data["status"] == "no_planes"


# ---------------------------------------------------------------------------
# FuturePlaneAnalyzer
# ---------------------------------------------------------------------------


def test_analyzer_creation() -> None:
    analyzer = FuturePlaneAnalyzer()

    assert analyzer.target_overlap_floor == pytest.approx(
        1.0
    )
    assert analyzer.safety_threshold == pytest.approx(
        0.75
    )
    assert (
        analyzer.missing_observation_penalty
        == pytest.approx(0.75)
    )
    assert analyzer.non_fonit_required is True


def test_analyzer_is_immutable() -> None:
    analyzer = FuturePlaneAnalyzer()

    with pytest.raises(FrozenInstanceError):
        analyzer.safety_threshold = 0.5


@pytest.mark.parametrize(
    "field_name",
    (
        "target_overlap_floor",
        "safety_threshold",
        "missing_observation_penalty",
    ),
)
def test_analyzer_rejects_invalid_unit_fields(
    field_name: str,
) -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match=field_name,
    ):
        FuturePlaneAnalyzer(
            **{
                field_name: 1.01,
            }
        )


def test_analyzer_rejects_nonboolean_non_fonit() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="non_fonit_required must be a boolean",
    ):
        FuturePlaneAnalyzer(
            non_fonit_required=1,
        )


def test_analyze_contribution_applies_matching_plane() -> None:
    analyzer = FuturePlaneAnalyzer()
    contribution = analyzer.analyze_contribution(
        make_plane(),
        make_candidate(),
    )

    assert contribution.applied is True
    assert contribution.veto_reason is None
    assert contribution.effective_multiplier > 1.0


def test_analyze_contribution_attenuates() -> None:
    analyzer = FuturePlaneAnalyzer()
    contribution = analyzer.analyze_contribution(
        make_plane(
            signed_effect=-0.50,
            direction=FuturePlaneDirection.ATTENUATE,
        ),
        make_candidate(),
    )

    assert contribution.applied is True
    assert 0.0 < contribution.effective_multiplier < 1.0


def test_analyze_contribution_vetoes_non_fonit_plane() -> None:
    contribution = FuturePlaneAnalyzer().analyze_contribution(
        make_plane(
            non_fonit_allowed=False,
        ),
        make_candidate(),
    )

    assert contribution.applied is False
    assert contribution.veto_reason == "non_fonit_gate"
    assert contribution.effective_multiplier == pytest.approx(
        1.0
    )


def test_analyze_contribution_vetoes_high_safety_risk() -> None:
    contribution = FuturePlaneAnalyzer(
        safety_threshold=0.20,
    ).analyze_contribution(
        make_plane(
            operational_risk=1.0,
            cascade_risk=1.0,
            reversibility=0.0,
        ),
        make_candidate(),
    )

    assert contribution.applied is False
    assert contribution.veto_reason == "safety_threshold"


def test_analyze_contribution_marks_not_applicable() -> None:
    contribution = FuturePlaneAnalyzer().analyze_contribution(
        make_plane(
            target_ids=("node:Z",),
        ),
        make_candidate(
            affected_target_ids=("node:A",),
        ),
    )

    assert contribution.applied is False
    assert contribution.veto_reason == "not_applicable"


def test_inferred_plane_gets_observation_penalty() -> None:
    analyzer = FuturePlaneAnalyzer(
        missing_observation_penalty=0.50,
    )
    active = analyzer.analyze_contribution(
        make_plane(
            future_plane_id="active",
            status=FuturePlaneStatus.ACTIVE,
            observability=1.0,
        ),
        make_candidate(),
    )
    inferred = analyzer.analyze_contribution(
        make_plane(
            future_plane_id="inferred",
            status=FuturePlaneStatus.INFERRED,
            observability=1.0,
        ),
        make_candidate(),
    )

    assert active.observability_weight == pytest.approx(
        1.0
    )
    assert inferred.observability_weight == pytest.approx(
        0.50
    )
    assert (
        active.effective_multiplier
        >
        inferred.effective_multiplier
    )


def test_target_overlap_reduces_exponent() -> None:
    analyzer = FuturePlaneAnalyzer()
    full = analyzer.analyze_contribution(
        make_plane(
            future_plane_id="full",
            target_ids=("node:A",),
        ),
        make_candidate(
            affected_target_ids=("node:A",),
        ),
    )
    partial = analyzer.analyze_contribution(
        make_plane(
            future_plane_id="partial",
            target_ids=(
                "node:A",
                "node:B",
            ),
        ),
        make_candidate(
            affected_target_ids=("node:A",),
        ),
    )

    assert full.target_weight == pytest.approx(1.0)
    assert partial.target_weight == pytest.approx(0.5)
    assert (
        full.effective_multiplier
        >
        partial.effective_multiplier
    )


def test_analyze_candidate_multiplies_planes() -> None:
    candidate = make_candidate()
    first = make_plane(
        future_plane_id="first",
        plane_id="mechanical",
        signed_effect=0.30,
    )
    second = make_plane(
        future_plane_id="second",
        plane_id="thermal",
        signed_effect=0.20,
    )

    analysis = FuturePlaneAnalyzer().analyze_candidate(
        candidate,
        (first, second),
    )

    expected = (
        analysis.contributions[0].effective_multiplier
        * analysis.contributions[1].effective_multiplier
    )

    assert analysis.applied_plane_count == 2
    assert analysis.combined_multiplier == pytest.approx(
        expected
    )


def test_analyze_candidate_rejects_duplicate_plane_ids() -> None:
    plane = make_plane()

    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="future_plane_id values must be unique",
    ):
        FuturePlaneAnalyzer().analyze_candidate(
            make_candidate(),
            (plane, plane),
        )


def test_analyze_no_planes() -> None:
    result = FuturePlaneAnalyzer().analyze(
        (make_candidate(),),
        (),
    )

    assert (
        result.status
        is FuturePlaneAnalysisStatus.NO_PLANES
    )
    assert result.plane_count == 0


def test_analyze_all_vetoed() -> None:
    result = FuturePlaneAnalyzer().analyze(
        (make_candidate(),),
        (
            make_plane(
                non_fonit_allowed=False,
            ),
        ),
    )

    assert (
        result.status
        is FuturePlaneAnalysisStatus.ALL_VETOED
    )
    assert result.vetoed_plane_count == 1


def test_analyze_no_applicable_planes() -> None:
    result = FuturePlaneAnalyzer().analyze(
        (
            make_candidate(
                affected_target_ids=("node:A",),
            ),
        ),
        (
            make_plane(
                target_ids=("node:Z",),
            ),
        ),
    )

    assert (
        result.status
        is FuturePlaneAnalysisStatus.NO_APPLICABLE_PLANES
    )


def test_analyze_partial_with_inferred_plane() -> None:
    result = FuturePlaneAnalyzer().analyze(
        (make_candidate(),),
        (
            make_plane(
                status=FuturePlaneStatus.INFERRED,
            ),
        ),
    )

    assert (
        result.status
        is FuturePlaneAnalysisStatus.PARTIAL
    )


def test_analyze_completed() -> None:
    result = FuturePlaneAnalyzer().analyze(
        (
            make_candidate(
                candidate_id="candidate-a",
            ),
            make_candidate(
                candidate_id="candidate-b",
            ),
        ),
        (
            make_plane(),
        ),
        metadata={
            "experiment": "future-plane",
        },
    )

    assert (
        result.status
        is FuturePlaneAnalysisStatus.COMPLETED
    )
    assert len(result.candidate_analyses) == 2
    assert result.metadata["candidate_count"] == 2
    assert result.metadata["experiment"] == "future-plane"


def test_analyze_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="candidate_id values must be unique",
    ):
        FuturePlaneAnalyzer().analyze(
            (
                make_candidate(
                    candidate_id="duplicate",
                ),
                make_candidate(
                    candidate_id="duplicate",
                ),
            ),
            (make_plane(),),
        )


def test_build_influence_sets() -> None:
    analyzer = FuturePlaneAnalyzer()
    result = analyzer.analyze(
        (
            make_candidate(
                candidate_id="candidate-a",
            ),
            make_candidate(
                candidate_id="candidate-b",
            ),
        ),
        (
            make_plane(),
        ),
    )

    influence_sets = analyzer.build_influence_sets(
        result
    )

    assert set(influence_sets) == {
        "candidate-a",
        "candidate-b",
    }
    assert all(
        isinstance(
            value,
            PlaneInfluenceSet,
        )
        for value in influence_sets.values()
    )


def test_build_influence_sets_is_read_only() -> None:
    analyzer = FuturePlaneAnalyzer()
    result = analyzer.analyze(
        (make_candidate(),),
        (make_plane(),),
    )

    influence_sets = analyzer.build_influence_sets(
        result
    )

    with pytest.raises(TypeError):
        influence_sets["new"] = PlaneInfluenceSet()


def test_build_influence_sets_rejects_invalid_result() -> None:
    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="result must be a FuturePlaneAnalysisResult",
    ):
        FuturePlaneAnalyzer().build_influence_sets(
            "result"
        )


def test_analyzer_round_trip() -> None:
    analyzer = FuturePlaneAnalyzer(
        target_overlap_floor=0.80,
        safety_threshold=0.60,
        missing_observation_penalty=0.50,
        non_fonit_required=False,
    )

    restored = FuturePlaneAnalyzer.from_dict(
        analyzer.to_dict()
    )

    assert restored == analyzer


def test_analyzer_from_dict_rejects_version() -> None:
    data = FuturePlaneAnalyzer().to_dict()
    data["version"] = "99.0"

    with pytest.raises(
        FuturePlaneAnalyzerError,
        match="unsupported future-plane analyzer version",
    ):
        FuturePlaneAnalyzer.from_dict(data)
