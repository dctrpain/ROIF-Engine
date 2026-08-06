"""
Role-separation tests for ROIF Root Detector.

These tests protect the semantic distinction between:

    D_origin
    D_fast
    D_root
    Node*

D_root represents an amplifying structural mediation channel.
Node* represents independent targeted intervention utility.
"""

from __future__ import annotations

import inspect

import pytest

from roif.roif_root_detector import (
    CandidateIntervention,
    InterventionKind,
    InterventionPolicy,
    RootDetectionMode,
    RootDetectorWeights,
    mediation_balance,
    mediation_throughput,
    node_star_score,
    propagation_gain,
    root_score,
)


# ============================================================================
# Helpers
# ============================================================================


def make_intervention(
    *,
    relative_reduction: float,
    collateral_effect: float = 0.0,
    intervention_cost: float = 0.0,
    safety_risk: float = 0.0,
    uncertainty: float = 0.0,
    irreversibility: float = 0.0,
    safe: bool = True,
) -> CandidateIntervention:
    """Create a minimal valid counterfactual intervention."""

    kind = next(iter(InterventionKind))

    baseline_burden = 1.0
    counterfactual_burden = max(
        0.0,
        baseline_burden - relative_reduction,
    )

    return CandidateIntervention(
        candidate_id="candidate",
        kind=kind,
        baseline_burden=baseline_burden,
        counterfactual_burden=counterfactual_burden,
        cascade_reduction=max(
            0.0,
            baseline_burden - counterfactual_burden,
        ),
        relative_reduction=relative_reduction,
        baseline_spectral_radius=1.0,
        counterfactual_spectral_radius=1.0,
        spectral_gain=0.0,
        affected_channel_count=1,
        collateral_effect=collateral_effect,
        intervention_cost=intervention_cost,
        safety_risk=safety_risk,
        uncertainty=uncertainty,
        irreversibility=irreversibility,
        safe=safe,
        metadata={},
    )


def calculate_root_score(
    *,
    incoming: float,
    outgoing: float,
    sensitivity: float = 0.0,
    reach: float = 1.0,
    exposure: float = 0.0,
    early_activity: float = 0.0,
    plane: float = 0.0,
    geometry: float = 0.0,
    material: float = 0.0,
    history: float = 0.0,
    mode: RootDetectionMode = RootDetectionMode.HYBRID,
) -> float:
    """Calculate a D_root score with default detector weights."""

    return root_score(
        incoming_strength_value=incoming,
        outgoing_strength_value=outgoing,
        tensor_sensitivity_value=sensitivity,
        downstream_reach_value=reach,
        exposure_value=exposure,
        early_activity_value=early_activity,
        plane_contribution_value=plane,
        geometry_deficit_value=geometry,
        material_deficit_value=material,
        history_effect_value=history,
        weights=RootDetectorWeights(),
        mode=mode,
    )


# ============================================================================
# Mediation balance
# ============================================================================


def test_mediation_balance_is_zero_for_pure_source() -> None:
    assert mediation_balance(0.0, 1.0) == pytest.approx(0.0)


def test_mediation_balance_is_zero_for_terminal_node() -> None:
    assert mediation_balance(1.0, 0.0) == pytest.approx(0.0)


def test_mediation_balance_is_maximal_for_balanced_channel() -> None:
    assert mediation_balance(1.0, 1.0) == pytest.approx(1.0)


def test_mediation_balance_is_scale_invariant() -> None:
    small = mediation_balance(0.2, 0.2)
    large = mediation_balance(20.0, 20.0)

    assert small == pytest.approx(large)
    assert small == pytest.approx(1.0)


def test_mediation_balance_rejects_directional_imbalance() -> None:
    balanced = mediation_balance(1.0, 1.0)
    imbalanced = mediation_balance(1.0, 0.1)

    assert balanced > imbalanced


def test_mediation_balance_clamps_negative_strengths() -> None:
    assert mediation_balance(-1.0, 1.0) == pytest.approx(0.0)
    assert mediation_balance(1.0, -1.0) == pytest.approx(0.0)


# ============================================================================
# Mediation throughput
# ============================================================================


def test_mediation_throughput_requires_incoming_and_outgoing_strength() -> None:
    assert mediation_throughput(0.0, 1.0) == pytest.approx(0.0)
    assert mediation_throughput(1.0, 0.0) == pytest.approx(0.0)


def test_mediation_throughput_preserves_structural_magnitude() -> None:
    weak = mediation_throughput(0.2, 0.2)
    strong = mediation_throughput(1.0, 1.0)

    assert weak == pytest.approx(0.2)
    assert strong == pytest.approx(1.0)
    assert strong > weak


def test_mediation_throughput_uses_geometric_mean() -> None:
    assert mediation_throughput(0.25, 1.0) == pytest.approx(0.5)


def test_mediation_throughput_clamps_negative_strengths() -> None:
    assert mediation_throughput(-1.0, 1.0) == pytest.approx(0.0)


# ============================================================================
# Propagation gain
# ============================================================================


def test_propagation_gain_identifies_attenuation() -> None:
    assert propagation_gain(1.0, 0.25) < 1.0


def test_propagation_gain_identifies_neutral_transmission() -> None:
    assert propagation_gain(1.0, 1.0) == pytest.approx(1.0)


def test_propagation_gain_identifies_amplification() -> None:
    assert propagation_gain(0.25, 1.0) > 1.0


def test_propagation_gain_is_capped() -> None:
    assert propagation_gain(
        0.001,
        100.0,
        maximum_gain=4.0,
    ) == pytest.approx(4.0)


def test_propagation_gain_is_zero_without_outgoing_strength() -> None:
    assert propagation_gain(1.0, 0.0) == pytest.approx(0.0)


# ============================================================================
# D_root semantics
# ============================================================================


def test_pure_source_cannot_win_as_structural_mediator() -> None:
    pure_source = calculate_root_score(
        incoming=0.0,
        outgoing=1.0,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    mediator = calculate_root_score(
        incoming=0.5,
        outgoing=0.5,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    assert pure_source == pytest.approx(0.0)
    assert mediator > pure_source


def test_terminal_node_cannot_win_as_structural_mediator() -> None:
    terminal = calculate_root_score(
        incoming=1.0,
        outgoing=0.0,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    mediator = calculate_root_score(
        incoming=0.5,
        outgoing=0.5,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    assert terminal == pytest.approx(0.0)
    assert mediator > terminal


def test_amplifying_mediator_outscores_attenuating_channel() -> None:
    attenuating = calculate_root_score(
        incoming=1.0,
        outgoing=0.2,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    amplifying = calculate_root_score(
        incoming=0.2,
        outgoing=1.0,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    assert amplifying > attenuating


def test_root_score_does_not_depend_on_origin_or_failure_features() -> None:
    baseline = calculate_root_score(
        incoming=0.4,
        outgoing=0.8,
        sensitivity=0.5,
        reach=0.75,
        exposure=0.0,
        early_activity=0.0,
        plane=0.0,
        geometry=0.0,
        material=0.0,
        history=0.0,
    )

    altered = calculate_root_score(
        incoming=0.4,
        outgoing=0.8,
        sensitivity=0.5,
        reach=0.75,
        exposure=1.0,
        early_activity=1.0,
        plane=1.0,
        geometry=1.0,
        material=1.0,
        history=1.0,
    )

    assert altered == pytest.approx(baseline)


def test_root_score_increases_with_tensor_sensitivity() -> None:
    low = calculate_root_score(
        incoming=0.5,
        outgoing=1.0,
        sensitivity=0.1,
        reach=1.0,
    )

    high = calculate_root_score(
        incoming=0.5,
        outgoing=1.0,
        sensitivity=0.9,
        reach=1.0,
    )

    assert high > low


def test_root_score_increases_with_downstream_reach() -> None:
    low = calculate_root_score(
        incoming=0.5,
        outgoing=1.0,
        sensitivity=0.0,
        reach=0.25,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    high = calculate_root_score(
        incoming=0.5,
        outgoing=1.0,
        sensitivity=0.0,
        reach=1.0,
        mode=RootDetectionMode.VIRTUAL_RESTORATION,
    )

    assert high > low


# ============================================================================
# Node* semantics
# ============================================================================


def test_node_star_signature_has_no_root_score_dependency() -> None:
    signature = inspect.signature(node_star_score)

    assert "root_score_value" not in signature.parameters


def test_node_star_prefers_greater_retained_gain() -> None:
    weights = RootDetectorWeights()

    stronger = make_intervention(
        relative_reduction=0.20,
        collateral_effect=0.10,
    )

    weaker = make_intervention(
        relative_reduction=0.05,
        collateral_effect=0.10,
    )

    stronger_score = node_star_score(
        intervention=stronger,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    weaker_score = node_star_score(
        intervention=weaker,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert stronger_score > weaker_score


def test_node_star_collateral_effect_attenuates_gain() -> None:
    weights = RootDetectorWeights()

    targeted = make_intervention(
        relative_reduction=0.20,
        collateral_effect=0.0,
    )

    diffuse = make_intervention(
        relative_reduction=0.20,
        collateral_effect=0.75,
    )

    targeted_score = node_star_score(
        intervention=targeted,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    diffuse_score = node_star_score(
        intervention=diffuse,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert targeted_score > diffuse_score


def test_node_star_full_collateral_removes_retained_gain() -> None:
    intervention = make_intervention(
        relative_reduction=0.50,
        collateral_effect=1.0,
    )

    score = node_star_score(
        intervention=intervention,
        confidence=1.0,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert score == pytest.approx(0.0)


def test_node_star_applies_intervention_cost_penalty() -> None:
    weights = RootDetectorWeights()

    low_cost = make_intervention(
        relative_reduction=0.20,
        intervention_cost=0.0,
    )

    high_cost = make_intervention(
        relative_reduction=0.20,
        intervention_cost=0.5,
    )

    low_cost_score = node_star_score(
        intervention=low_cost,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    high_cost_score = node_star_score(
        intervention=high_cost,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert low_cost_score > high_cost_score


def test_node_star_applies_uncertainty_penalty() -> None:
    weights = RootDetectorWeights()

    certain = make_intervention(
        relative_reduction=0.20,
        uncertainty=0.0,
    )

    uncertain = make_intervention(
        relative_reduction=0.20,
        uncertainty=0.5,
    )

    certain_score = node_star_score(
        intervention=certain,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    uncertain_score = node_star_score(
        intervention=uncertain,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert certain_score > uncertain_score


def test_node_star_confidence_scales_score() -> None:
    intervention = make_intervention(
        relative_reduction=0.20,
    )

    full_confidence = node_star_score(
        intervention=intervention,
        confidence=1.0,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.ADAPTIVE,
    )

    half_confidence = node_star_score(
        intervention=intervention,
        confidence=0.5,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert half_confidence == pytest.approx(
        0.5 * full_confidence
    )


def test_strict_policy_strongly_rejects_unsafe_intervention() -> None:
    intervention = make_intervention(
        relative_reduction=0.90,
        safe=False,
    )

    score = node_star_score(
        intervention=intervention,
        confidence=1.0,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.STRICT_LINEAR,
    )

    assert score <= -1.0


def test_adaptive_policy_penalizes_unsafe_intervention() -> None:
    intervention = make_intervention(
        relative_reduction=0.20,
        safe=False,
    )

    safe_equivalent = make_intervention(
        relative_reduction=0.20,
        safe=True,
    )

    unsafe_score = node_star_score(
        intervention=intervention,
        confidence=1.0,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.ADAPTIVE,
    )

    safe_score = node_star_score(
        intervention=safe_equivalent,
        confidence=1.0,
        weights=RootDetectorWeights(),
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert unsafe_score < safe_score


def test_node_star_matches_foot_knee_retained_gain_order() -> None:
    """
    Protect the mathematical ordering discovered during clinical validation.

    This test contains no clinical label and does not run the validation case.
    It tests only the generic counterfactual utility relationship.
    """

    weights = RootDetectorWeights()

    forefoot_compliance = make_intervention(
        relative_reduction=0.023218175202344617,
        collateral_effect=0.0327593364037825,
    )

    tibial_rotation_control = make_intervention(
        relative_reduction=0.006482789596588138,
        collateral_effect=0.006505987128377641,
    )

    compliance_score = node_star_score(
        intervention=forefoot_compliance,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    rotation_score = node_star_score(
        intervention=tibial_rotation_control,
        confidence=1.0,
        weights=weights,
        policy=InterventionPolicy.ADAPTIVE,
    )

    assert compliance_score > rotation_score
