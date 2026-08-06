"""
Tests for the universal Active Probe Planner.

The planner ranks policy-admissible Probe definitions and selects Probe*.
It must not execute Probes, mutate graphs, or call the ROIF Solver.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.probe_entities import (
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
    ProbeRegime,
)
from roif.probe_planner import (
    NoAdmissibleProbeError,
    ProbeCandidateStatus,
    ProbeInformationEstimate,
    ProbePlanCandidate,
    ProbePlanner,
    ProbePlannerConfig,
    ProbePlannerError,
    ProbePlannerWeights,
    ProbePlanResult,
    ProbeScoreComponents,
    ProbeSelectionStatus,
    calculate_probe_score,
)
from roif.probe_policy import (
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
    ProbePolicyDecision,
    ProbeRiskLevel,
)
from roif.probe_registry import (
    ProbeQuery,
    ProbeRegistry,
)


# ============================================================================
# Helpers
# ============================================================================


def make_probe(
    identifier: str,
    *,
    method: ProbeMethod = ProbeMethod.OBSERVATION,
    regime: ProbeRegime = ProbeRegime.STATIC,
    purpose: ProbePurpose = ProbePurpose.REDUCE_UNCERTAINTY,
    perturbation: PerturbationType = PerturbationType.NONE,
    tags: tuple[str, ...] = (),
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name=identifier,
        method=method,
        regime=regime,
        purpose=purpose,
        perturbation=Perturbation(kind=perturbation),
        description=f"Probe definition for {identifier}.",
        tags=tags,
    )


def make_estimate(
    identifier: str,
    *,
    information_gain: float = 0.5,
    uncertainty_reduction: float = 0.5,
    hypothesis_discrimination: float = 0.5,
    graph_coverage: float = 0.5,
    novelty: float = 0.5,
    feasibility: float = 1.0,
    confidence: float = 1.0,
    redundancy: float = 0.0,
    disruption: float = 0.0,
) -> ProbeInformationEstimate:
    return ProbeInformationEstimate(
        probe_identifier=identifier,
        information_gain=information_gain,
        uncertainty_reduction=uncertainty_reduction,
        hypothesis_discrimination=hypothesis_discrimination,
        graph_coverage=graph_coverage,
        novelty=novelty,
        feasibility=feasibility,
        confidence=confidence,
        redundancy=redundancy,
        disruption=disruption,
    )


@pytest.fixture
def observation_probe() -> ProbeDefinition:
    return make_probe(
        "probe.observation",
        method=ProbeMethod.OBSERVATION,
        regime=ProbeRegime.REST,
        tags=("passive", "universal"),
    )


@pytest.fixture
def load_probe() -> ProbeDefinition:
    return make_probe(
        "probe.load",
        method=ProbeMethod.INSTRUMENTAL,
        perturbation=PerturbationType.LOAD_CHANGE,
        purpose=ProbePurpose.REJECT_HYPOTHESIS,
        tags=("controlled", "load"),
    )


@pytest.fixture
def simulation_probe() -> ProbeDefinition:
    return make_probe(
        "probe.simulation",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.TRANSITION,
        perturbation=PerturbationType.SIMULATED_INTERVENTION,
        tags=("simulation", "counterfactual"),
    )


@pytest.fixture
def registry(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> ProbeRegistry:
    return ProbeRegistry(
        (
            observation_probe,
            load_probe,
            simulation_probe,
        )
    )


@pytest.fixture
def estimates() -> tuple[ProbeInformationEstimate, ...]:
    return (
        make_estimate(
            "probe.observation",
            information_gain=0.20,
            uncertainty_reduction=0.25,
            hypothesis_discrimination=0.10,
            graph_coverage=0.20,
            novelty=0.10,
        ),
        make_estimate(
            "probe.load",
            information_gain=0.90,
            uncertainty_reduction=0.85,
            hypothesis_discrimination=0.95,
            graph_coverage=0.80,
            novelty=0.75,
        ),
        make_estimate(
            "probe.simulation",
            information_gain=0.60,
            uncertainty_reduction=0.70,
            hypothesis_discrimination=0.65,
            graph_coverage=0.60,
            novelty=0.50,
        ),
    )


# ============================================================================
# ProbeInformationEstimate
# ============================================================================


def test_information_estimate_normalizes_identifier() -> None:
    estimate = make_estimate("  probe.test  ")

    assert estimate.probe_identifier == "probe.test"


@pytest.mark.parametrize(
    "field_name",
    (
        "information_gain",
        "uncertainty_reduction",
        "hypothesis_discrimination",
        "graph_coverage",
        "novelty",
        "feasibility",
        "confidence",
        "redundancy",
        "disruption",
    ),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_information_estimate_rejects_out_of_range_value(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ProbeInformationEstimate(
            probe_identifier="probe.test",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "information_gain",
        "uncertainty_reduction",
        "hypothesis_discrimination",
        "graph_coverage",
        "novelty",
        "feasibility",
        "confidence",
        "redundancy",
        "disruption",
    ),
)
def test_information_estimate_rejects_bool(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProbeInformationEstimate(
            probe_identifier="probe.test",
            **{field_name: True},
        )


def test_information_estimate_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError):
        ProbeInformationEstimate(probe_identifier=" ")


def test_information_estimate_metadata_is_immutable() -> None:
    estimate = ProbeInformationEstimate(
        probe_identifier="probe.test",
        metadata={"source": "simulation"},
    )

    assert isinstance(estimate.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        estimate.metadata["source"] = "changed"  # type: ignore[index]


# ============================================================================
# Weights and configuration
# ============================================================================


def test_default_weights_are_nonnegative() -> None:
    weights = ProbePlannerWeights()

    for field_name in weights.__dataclass_fields__:
        assert getattr(weights, field_name) >= 0.0


def test_weights_reject_negative_value() -> None:
    with pytest.raises(ValueError):
        ProbePlannerWeights(information_gain=-1.0)


def test_weights_reject_bool() -> None:
    with pytest.raises(TypeError):
        ProbePlannerWeights(novelty=True)  # type: ignore[arg-type]


def test_config_defaults() -> None:
    config = ProbePlannerConfig()

    assert config.include_authorization_required is True
    assert config.allow_missing_estimates is False
    assert config.raise_when_no_admissible_probe is False
    assert config.maximum_results is None
    assert config.minimum_score is None


def test_config_rejects_wrong_weights_type() -> None:
    with pytest.raises(TypeError):
        ProbePlannerConfig(weights=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    (
        "include_authorization_required",
        "allow_missing_estimates",
        "raise_when_no_admissible_probe",
    ),
)
def test_config_boolean_fields_require_bool(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProbePlannerConfig(**{field_name: 1})


def test_config_rejects_zero_cost_normalization() -> None:
    with pytest.raises(ValueError):
        ProbePlannerConfig(cost_normalization=0.0)


def test_config_rejects_zero_duration_normalization() -> None:
    with pytest.raises(ValueError):
        ProbePlannerConfig(
            duration_normalization_seconds=0.0
        )


@pytest.mark.parametrize("value", (0, -1))
def test_config_rejects_invalid_maximum_results(
    value: int,
) -> None:
    with pytest.raises(ValueError):
        ProbePlannerConfig(maximum_results=value)


def test_config_rejects_bool_maximum_results() -> None:
    with pytest.raises(TypeError):
        ProbePlannerConfig(maximum_results=True)  # type: ignore[arg-type]


# ============================================================================
# Score calculation
# ============================================================================


def test_score_is_information_value_minus_penalties() -> None:
    estimate = make_estimate(
        "probe.test",
        information_gain=1.0,
        uncertainty_reduction=0.0,
        hypothesis_discrimination=0.0,
        graph_coverage=0.0,
        novelty=0.0,
    )

    config = ProbePlannerConfig(
        weights=ProbePlannerWeights(
            information_gain=1.0,
            uncertainty_reduction=0.0,
            hypothesis_discrimination=0.0,
            graph_coverage=0.0,
            novelty=0.0,
            expected_cost=1.0,
            expected_duration=0.0,
            uncertainty=0.0,
            cascade_risk=0.0,
            redundancy=0.0,
            disruption=0.0,
        ),
        cost_normalization=10.0,
    )

    components = calculate_probe_score(
        estimate,
        ProbeAssessment(expected_cost=5.0),
        config,
    )

    assert components.information_value == pytest.approx(1.0)
    assert components.cost_penalty == pytest.approx(0.5)
    assert components.final_score == pytest.approx(0.5)


def test_feasibility_scales_gross_score() -> None:
    full = calculate_probe_score(
        make_estimate("probe.full", feasibility=1.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    half = calculate_probe_score(
        make_estimate("probe.half", feasibility=0.5),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    assert half.gross_score == pytest.approx(
        0.5 * full.gross_score
    )


def test_estimate_confidence_scales_gross_score() -> None:
    full = calculate_probe_score(
        make_estimate("probe.full", confidence=1.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    half = calculate_probe_score(
        make_estimate("probe.half", confidence=0.5),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    assert half.gross_score == pytest.approx(
        0.5 * full.gross_score
    )


def test_cost_penalty_is_capped_by_normalization() -> None:
    config = ProbePlannerConfig(
        weights=ProbePlannerWeights(expected_cost=0.35),
        cost_normalization=10.0,
    )

    components = calculate_probe_score(
        make_estimate("probe.test"),
        ProbeAssessment(expected_cost=1000.0),
        config,
    )

    assert components.cost_penalty == pytest.approx(0.35)


def test_duration_penalty_is_capped_by_normalization() -> None:
    config = ProbePlannerConfig(
        weights=ProbePlannerWeights(expected_duration=0.20),
        duration_normalization_seconds=10.0,
    )

    components = calculate_probe_score(
        make_estimate("probe.test"),
        ProbeAssessment(expected_duration_seconds=1000.0),
        config,
    )

    assert components.duration_penalty == pytest.approx(0.20)


def test_redundancy_reduces_score() -> None:
    low = calculate_probe_score(
        make_estimate("probe.low", redundancy=0.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    high = calculate_probe_score(
        make_estimate("probe.high", redundancy=1.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    assert low.final_score > high.final_score


def test_disruption_reduces_score() -> None:
    low = calculate_probe_score(
        make_estimate("probe.low", disruption=0.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    high = calculate_probe_score(
        make_estimate("probe.high", disruption=1.0),
        ProbeAssessment(),
        ProbePlannerConfig(),
    )

    assert low.final_score > high.final_score


def test_cascade_risk_reduces_score() -> None:
    estimate = make_estimate("probe.test")

    low = calculate_probe_score(
        estimate,
        ProbeAssessment(cascade_risk=0.0),
        ProbePlannerConfig(),
    )

    high = calculate_probe_score(
        estimate,
        ProbeAssessment(cascade_risk=1.0),
        ProbePlannerConfig(),
    )

    assert low.final_score > high.final_score


def test_calculate_score_rejects_wrong_inputs() -> None:
    estimate = make_estimate("probe.test")
    assessment = ProbeAssessment()
    config = ProbePlannerConfig()

    with pytest.raises(TypeError):
        calculate_probe_score(
            object(),  # type: ignore[arg-type]
            assessment,
            config,
        )

    with pytest.raises(TypeError):
        calculate_probe_score(
            estimate,
            object(),  # type: ignore[arg-type]
            config,
        )

    with pytest.raises(TypeError):
        calculate_probe_score(
            estimate,
            assessment,
            object(),  # type: ignore[arg-type]
        )


# ============================================================================
# Planner construction
# ============================================================================


def test_planner_accepts_registry(
    registry: ProbeRegistry,
) -> None:
    planner = ProbePlanner(registry)

    assert planner.registry is registry
    assert isinstance(planner.policy, ProbePolicy)
    assert isinstance(planner.config, ProbePlannerConfig)


def test_planner_accepts_registry_snapshot(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()
    planner = ProbePlanner(snapshot)

    assert planner.registry is snapshot


def test_planner_rejects_wrong_registry() -> None:
    with pytest.raises(TypeError):
        ProbePlanner(object())  # type: ignore[arg-type]


def test_planner_rejects_wrong_policy(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(
            registry,
            policy=object(),  # type: ignore[arg-type]
        )


def test_planner_rejects_wrong_config(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(
            registry,
            config=object(),  # type: ignore[arg-type]
        )


# ============================================================================
# Basic selection
# ============================================================================


def test_planner_selects_highest_scoring_probe(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    result = ProbePlanner(registry).plan(estimates)

    assert result.status is ProbeSelectionStatus.SELECTED
    assert result.probe_star is load_probe
    assert result.selected is not None
    assert result.selected.rank == 1


def test_rank_values_are_consecutive(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    result = ProbePlanner(registry).plan(estimates)

    assert tuple(
        candidate.rank
        for candidate in result.ranked_candidates
    ) == (1, 2, 3)


def test_ranked_scores_are_descending(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    result = ProbePlanner(registry).plan(estimates)

    scores = tuple(
        candidate.score
        for candidate in result.ranked_candidates
    )

    assert scores == tuple(
        sorted(scores, reverse=True)
    )


def test_plan_is_deterministic(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    planner = ProbePlanner(registry)

    first = planner.plan(estimates)
    second = planner.plan(estimates)

    assert first == second


# ============================================================================
# Policy integration and safety
# ============================================================================


def test_rejected_probe_is_excluded_from_ranking(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    policy = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(estimates)

    ranked_ids = {
        candidate.definition.identifier
        for candidate in result.ranked_candidates
    }

    assert load_probe.identifier not in ranked_ids
    assert any(
        candidate.definition is load_probe
        for candidate in result.rejected_candidates
    )


def test_safety_veto_beats_information_gain(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    assessments = {
        load_probe.identifier: ProbeAssessment(
            risk_level=ProbeRiskLevel.CRITICAL,
        )
    }

    result = ProbePlanner(registry).plan(
        estimates,
        assessments=assessments,
        context=ProbePolicyContext(
            maximum_risk_level=ProbeRiskLevel.LOW
        ),
    )

    assert result.probe_star is not load_probe


def test_non_fonit_veto_beats_information_gain(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    assessments = {
        load_probe.identifier: ProbeAssessment(
            uncontrolled_propagation_possible=True
        )
    }

    result = ProbePlanner(registry).plan(
        estimates,
        assessments=assessments,
    )

    assert result.probe_star is not load_probe
    assert any(
        candidate.definition is load_probe
        and candidate.status is ProbeCandidateStatus.REJECTED
        for candidate in result.rejected_candidates
    )


def test_policy_rejection_is_recorded(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    policy = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.SIMULATION}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(estimates)

    rejected = result.rejected_candidates

    assert len(rejected) == 1
    assert (
        rejected[0].policy_result.decision
        is ProbePolicyDecision.REJECT
    )


# ============================================================================
# Authorization
# ============================================================================


def test_authorization_required_candidate_can_be_ranked(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(estimates)

    assert (
        result.status
        is ProbeSelectionStatus.AUTHORIZATION_REQUIRED
    )
    assert result.selected is None
    assert result.probe_star is None
    assert result.ranked_candidates[0].definition is load_probe


def test_human_authorization_makes_candidate_admissible(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(
        estimates,
        context=ProbePolicyContext(human_authorized=True),
    )

    assert result.status is ProbeSelectionStatus.SELECTED
    assert result.probe_star is load_probe


def test_authorization_candidates_can_be_excluded(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    load_probe: ProbeDefinition,
) -> None:
    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    planner = ProbePlanner(
        registry,
        policy=policy,
        config=ProbePlannerConfig(
            include_authorization_required=False
        ),
    )

    result = planner.plan(estimates)

    assert result.status is ProbeSelectionStatus.SELECTED
    assert result.probe_star is not load_probe


def test_admissible_candidate_wins_equal_score_tie_over_authorization(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry((observation_probe, load_probe))

    equal_estimates = (
        make_estimate(observation_probe.identifier),
        make_estimate(load_probe.identifier),
    )

    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(equal_estimates)

    assert result.probe_star is observation_probe


# ============================================================================
# Missing estimates
# ============================================================================


def test_missing_estimate_candidate_is_recorded(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    result = ProbePlanner(registry).plan(
        (
            make_estimate("probe.load"),
            make_estimate("probe.simulation"),
        )
    )

    assert len(result.missing_estimate_candidates) == 1
    assert (
        result.missing_estimate_candidates[0].definition
        is observation_probe
    )


def test_missing_estimate_is_not_ranked_by_default(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    result = ProbePlanner(registry).plan(
        (
            make_estimate("probe.load"),
            make_estimate("probe.simulation"),
        )
    )

    assert all(
        candidate.definition is not observation_probe
        for candidate in result.ranked_candidates
    )


def test_missing_estimate_can_be_included_with_zero_confidence(
    registry: ProbeRegistry,
) -> None:
    planner = ProbePlanner(
        registry,
        config=ProbePlannerConfig(
            allow_missing_estimates=True
        ),
    )

    result = planner.plan(())

    assert len(result.ranked_candidates) == 3
    assert len(result.missing_estimate_candidates) == 3
    assert all(
        candidate.score is not None
        for candidate in result.ranked_candidates
    )


def test_empty_estimates_produce_no_admissible_probes_by_default(
    registry: ProbeRegistry,
) -> None:
    result = ProbePlanner(registry).plan(())

    assert (
        result.status
        is ProbeSelectionStatus.NO_ADMISSIBLE_PROBES
    )
    assert result.selected is None


# ============================================================================
# Estimate input validation
# ============================================================================


def test_estimate_mapping_is_supported(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    estimate_map = {
        estimate.probe_identifier: estimate
        for estimate in estimates
    }

    result = ProbePlanner(registry).plan(estimate_map)

    assert result.status is ProbeSelectionStatus.SELECTED


def test_estimate_mapping_key_must_match_estimate_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ProbePlannerError):
        ProbePlanner(registry).plan(
            {
                "probe.wrong": make_estimate("probe.load"),
            }
        )


def test_duplicate_estimates_in_iterable_are_rejected(
    registry: ProbeRegistry,
) -> None:
    estimate = make_estimate("probe.load")

    with pytest.raises(ProbePlannerError):
        ProbePlanner(registry).plan(
            (estimate, estimate)
        )


def test_non_estimate_iterable_member_is_rejected(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            (object(),)  # type: ignore[arg-type]
        )


def test_string_estimates_are_rejected(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            "probe.load"  # type: ignore[arg-type]
        )


def test_assessments_must_be_mapping(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            estimates,
            assessments=(),  # type: ignore[arg-type]
        )


def test_assessment_values_must_be_probe_assessment(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            estimates,
            assessments={
                "probe.load": object(),  # type: ignore[dict-item]
            },
        )


def test_context_type_is_validated(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            estimates,
            context=object(),  # type: ignore[arg-type]
        )


# ============================================================================
# Query filtering
# ============================================================================


def test_query_filters_registry_before_planning(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
    simulation_probe: ProbeDefinition,
) -> None:
    result = ProbePlanner(registry).plan(
        estimates,
        query=ProbeQuery(
            method=ProbeMethod.SIMULATION
        ),
    )

    assert result.probe_star is simulation_probe
    assert len(result.ranked_candidates) == 1


def test_query_can_produce_no_candidates(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    result = ProbePlanner(registry).plan(
        estimates,
        query=ProbeQuery(
            required_tags=frozenset({"missing"})
        ),
    )

    assert result.status is ProbeSelectionStatus.NO_CANDIDATES
    assert result.probe_star is None


def test_query_type_is_validated(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    with pytest.raises(TypeError):
        ProbePlanner(registry).plan(
            estimates,
            query=object(),  # type: ignore[arg-type]
        )


def test_query_works_with_registry_snapshot(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    planner = ProbePlanner(registry.snapshot())

    result = planner.plan(
        estimates,
        query=ProbeQuery(
            required_tags=frozenset({"simulation"})
        ),
    )

    assert result.probe_star is not None
    assert result.probe_star.identifier == "probe.simulation"


# ============================================================================
# Thresholds and result limits
# ============================================================================


def test_minimum_score_rejects_low_scoring_candidates(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    planner = ProbePlanner(
        registry,
        config=ProbePlannerConfig(
            minimum_score=100.0
        ),
    )

    result = planner.plan(estimates)

    assert (
        result.status
        is ProbeSelectionStatus.NO_ADMISSIBLE_PROBES
    )
    assert result.ranked_candidates == ()
    assert len(result.rejected_candidates) == 3


def test_maximum_results_limits_ranked_output(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    planner = ProbePlanner(
        registry,
        config=ProbePlannerConfig(maximum_results=2),
    )

    result = planner.plan(estimates)

    assert len(result.ranked_candidates) == 2
    assert result.selected is result.ranked_candidates[0]


def test_raise_when_no_admissible_probe(
    registry: ProbeRegistry,
) -> None:
    planner = ProbePlanner(
        registry,
        config=ProbePlannerConfig(
            raise_when_no_admissible_probe=True
        ),
    )

    with pytest.raises(NoAdmissibleProbeError):
        planner.plan(())


# ============================================================================
# Tie-breaking
# ============================================================================


def test_equal_scores_use_identifier_as_final_tiebreak() -> None:
    probe_b = make_probe("probe.b")
    probe_a = make_probe("probe.a")

    registry = ProbeRegistry((probe_b, probe_a))

    result = ProbePlanner(registry).plan(
        (
            make_estimate("probe.b"),
            make_estimate("probe.a"),
        )
    )

    assert tuple(
        candidate.definition.identifier
        for candidate in result.ranked_candidates
    ) == ("probe.a", "probe.b")


def test_equal_score_prefers_higher_estimate_confidence() -> None:
    probe_a = make_probe("probe.a")
    probe_b = make_probe("probe.b")

    registry = ProbeRegistry((probe_a, probe_b))

    high_confidence = make_estimate(
        "probe.b",
        information_gain=1.0,
        uncertainty_reduction=0.0,
        hypothesis_discrimination=0.0,
        graph_coverage=0.0,
        novelty=0.0,
        confidence=1.0,
    )

    lower_confidence = make_estimate(
        "probe.a",
        information_gain=1.0,
        uncertainty_reduction=0.0,
        hypothesis_discrimination=0.0,
        graph_coverage=0.0,
        novelty=0.0,
        feasibility=1.0,
        confidence=0.5,
    )

    # Adjust feasibility so both gross scores are equal.
    high_confidence = ProbeInformationEstimate(
        probe_identifier="probe.b",
        information_gain=0.5,
        feasibility=1.0,
        confidence=1.0,
    )

    lower_confidence = ProbeInformationEstimate(
        probe_identifier="probe.a",
        information_gain=1.0,
        feasibility=1.0,
        confidence=0.5,
    )

    result = ProbePlanner(registry).plan(
        (lower_confidence, high_confidence)
    )

    assert result.ranked_candidates[0].definition is probe_b


# ============================================================================
# Candidate and result contracts
# ============================================================================


def test_selected_candidate_is_selectable(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    result = ProbePlanner(registry).plan(estimates)

    assert result.selected is not None
    assert result.selected.selectable


def test_rejected_candidate_is_not_selectable(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    policy = ProbePolicy(
        forbidden_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )

    result = ProbePlanner(
        registry,
        policy=policy,
    ).plan(estimates)

    assert result.rejected_candidates
    assert all(
        not candidate.selectable
        for candidate in result.rejected_candidates
        if candidate.status is ProbeCandidateStatus.REJECTED
    )


def test_plan_result_metadata_is_immutable(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    base = ProbePlanner(registry).plan(estimates)

    result = ProbePlanResult(
        status=base.status,
        selected=base.selected,
        ranked_candidates=base.ranked_candidates,
        rejected_candidates=base.rejected_candidates,
        missing_estimate_candidates=(
            base.missing_estimate_candidates
        ),
        metadata={"source": "test"},
    )

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["source"] = "changed"  # type: ignore[index]


def test_selected_status_requires_selected_candidate() -> None:
    with pytest.raises(ValueError):
        ProbePlanResult(
            status=ProbeSelectionStatus.SELECTED,
            selected=None,
            ranked_candidates=(),
            rejected_candidates=(),
            missing_estimate_candidates=(),
        )


def test_nonselected_status_rejects_selected_candidate(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    selected = ProbePlanner(registry).plan(estimates).selected

    assert selected is not None

    with pytest.raises(ValueError):
        ProbePlanResult(
            status=ProbeSelectionStatus.NO_CANDIDATES,
            selected=selected,
            ranked_candidates=(),
            rejected_candidates=(),
            missing_estimate_candidates=(),
        )


def test_score_components_are_auditable(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    result = ProbePlanner(registry).plan(estimates)

    for candidate in result.ranked_candidates:
        assert isinstance(
            candidate.components,
            ProbeScoreComponents,
        )
        assert candidate.score == pytest.approx(
            candidate.components.final_score
        )


# ============================================================================
# Architectural boundaries
# ============================================================================


def test_planner_has_no_execution_solver_or_graph_mutation_methods(
    registry: ProbeRegistry,
) -> None:
    planner = ProbePlanner(registry)

    forbidden_methods = (
        "execute",
        "run_probe",
        "solve",
        "update_graph",
        "apply_graph_update",
        "calculate_d_origin",
        "calculate_d_fast",
        "calculate_d_root",
        "calculate_node_star",
    )

    for method_name in forbidden_methods:
        assert not hasattr(planner, method_name)


def test_planning_does_not_mutate_registry(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    before = registry.definitions()

    ProbePlanner(registry).plan(estimates)

    assert registry.definitions() == before


def test_planning_does_not_mutate_estimates(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    before = tuple(estimates)

    ProbePlanner(registry).plan(estimates)

    assert estimates == before


def test_planning_does_not_mutate_assessments(
    registry: ProbeRegistry,
    estimates: tuple[ProbeInformationEstimate, ...],
) -> None:
    assessment = ProbeAssessment(
        expected_cost=0.1,
        cascade_risk=0.1,
    )

    assessments = {"probe.load": assessment}

    ProbePlanner(registry).plan(
        estimates,
        assessments=assessments,
    )

    assert assessments == {"probe.load": assessment}
