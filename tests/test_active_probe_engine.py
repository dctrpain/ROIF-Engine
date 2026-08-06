"""
Tests for the universal Active Probe Engine lifecycle.

The engine coordinates planning and Probe execution records, but must not
physically perform perturbations, mutate graphs, call Solver, or calculate
D_origin, D_fast, D_root, or Node*.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.active_probe_engine import (
    ActiveProbeEngine,
    ActiveProbeEngineConfig,
    ActiveProbeEngineError,
    ActiveProbeEngineSnapshot,
    ActiveProbeEngineState,
    ActiveProbeRun,
    InvalidEngineStateError,
    ObservationDeltaMode,
    ObservationTransition,
    ProbeAuthorizationRequiredError,
    ProbeCompletionStatus,
    ProbeExecutionNotFoundError,
    ProbeExecutionRejectedError,
    ProbeObservationError,
    default_observation_delta,
)
from roif.probe_entities import (
    GraphUpdate,
    Observation,
    ObservationChannel,
    ObservationDelta,
    Perturbation,
    PerturbationType,
    ProbeDefinition,
    ProbeExecution,
    ProbeMethod,
    ProbePhase,
    ProbePurpose,
    ProbeRegime,
    ProbeResult,
)
from roif.probe_planner import (
    ProbeInformationEstimate,
    ProbePlannerConfig,
    ProbeSelectionStatus,
)
from roif.probe_policy import (
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
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
    identifier: str = "probe.controlled",
    *,
    method: ProbeMethod = ProbeMethod.INSTRUMENTAL,
    perturbation: PerturbationType = PerturbationType.LOAD_CHANGE,
    tags: tuple[str, ...] = ("controlled",),
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name=identifier,
        method=method,
        regime=ProbeRegime.STATIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=perturbation,
            magnitude=1.0,
            magnitude_units="relative",
            duration_seconds=1.0,
        ),
        description="Universal controlled experiment.",
        tags=tags,
    )


def make_estimate(
    identifier: str,
    *,
    information_gain: float = 0.8,
    uncertainty_reduction: float = 0.8,
    hypothesis_discrimination: float = 0.7,
    graph_coverage: float = 0.7,
    novelty: float = 0.6,
) -> ProbeInformationEstimate:
    return ProbeInformationEstimate(
        probe_identifier=identifier,
        information_gain=information_gain,
        uncertainty_reduction=uncertainty_reduction,
        hypothesis_discrimination=hypothesis_discrimination,
        graph_coverage=graph_coverage,
        novelty=novelty,
    )


def make_observation(
    phase: ProbePhase,
    *,
    value: object,
    channel: ObservationChannel = ObservationChannel.DISPLACEMENT,
    target: str = "node_A",
    units: str | None = "mm",
) -> Observation:
    return Observation(
        phase=phase,
        channel=channel,
        target=target,
        value=value,
        units=units,
        confidence=0.95,
    )


def add_numeric_pair(
    engine: ActiveProbeEngine,
    *,
    baseline: float = 1.0,
    reassessment: float = 1.5,
) -> tuple[Observation, Observation]:
    before = make_observation(
        ProbePhase.BASELINE,
        value=baseline,
    )
    after = make_observation(
        ProbePhase.REASSESSMENT,
        value=reassessment,
    )

    engine.add_observation(before)
    engine.add_observation(after)

    return before, after


@pytest.fixture
def controlled_probe() -> ProbeDefinition:
    return make_probe("probe.controlled")


@pytest.fixture
def passive_probe() -> ProbeDefinition:
    return make_probe(
        "probe.passive",
        method=ProbeMethod.OBSERVATION,
        perturbation=PerturbationType.NONE,
        tags=("passive",),
    )


@pytest.fixture
def registry(
    controlled_probe: ProbeDefinition,
    passive_probe: ProbeDefinition,
) -> ProbeRegistry:
    return ProbeRegistry(
        (
            controlled_probe,
            passive_probe,
        )
    )


@pytest.fixture
def engine(
    registry: ProbeRegistry,
) -> ActiveProbeEngine:
    return ActiveProbeEngine(registry)


# ============================================================================
# Default delta calculation
# ============================================================================


def test_default_numeric_delta() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value=2.0,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=5.5,
    )

    assert default_observation_delta(
        baseline,
        reassessment,
    ) == pytest.approx(3.5)


def test_default_numeric_delta_accepts_ints() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value=2,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=5,
    )

    assert default_observation_delta(
        baseline,
        reassessment,
    ) == pytest.approx(3.0)


def test_default_delta_returns_transition_for_text() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value="stable",
        units=None,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value="unstable",
        units=None,
    )

    delta = default_observation_delta(
        baseline,
        reassessment,
    )

    assert isinstance(delta, ObservationTransition)
    assert delta.before == "stable"
    assert delta.after == "unstable"
    assert delta.changed


def test_unchanged_transition_reports_not_changed() -> None:
    transition = ObservationTransition(
        before="stable",
        after="stable",
    )

    assert transition.changed is False


def test_numeric_only_mode_rejects_non_numeric_values() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value="before",
        units=None,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value="after",
        units=None,
    )

    with pytest.raises(ProbeObservationError):
        default_observation_delta(
            baseline,
            reassessment,
            mode=ObservationDeltaMode.NUMERIC_ONLY,
        )


def test_explicit_only_mode_rejects_automatic_text_delta() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value="before",
        units=None,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value="after",
        units=None,
    )

    with pytest.raises(ProbeObservationError):
        default_observation_delta(
            baseline,
            reassessment,
            mode=ObservationDeltaMode.EXPLICIT_ONLY,
        )


def test_explicit_only_mode_still_calculates_numeric_delta() -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=2.0,
    )

    assert default_observation_delta(
        baseline,
        reassessment,
        mode=ObservationDeltaMode.EXPLICIT_ONLY,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    (
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_default_delta_rejects_nonfinite_baseline(
    value: float,
) -> None:
    baseline = make_observation(
        ProbePhase.BASELINE,
        value=value,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=1.0,
    )

    with pytest.raises(ProbeObservationError):
        default_observation_delta(
            baseline,
            reassessment,
        )


def test_default_delta_validates_inputs() -> None:
    observation = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )

    with pytest.raises(TypeError):
        default_observation_delta(
            object(),  # type: ignore[arg-type]
            observation,
        )

    with pytest.raises(TypeError):
        default_observation_delta(
            observation,
            object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        default_observation_delta(
            observation,
            observation,
            mode="numeric",  # type: ignore[arg-type]
        )


# ============================================================================
# Configuration
# ============================================================================


def test_engine_config_defaults() -> None:
    config = ActiveProbeEngineConfig()

    assert config.require_baseline_observation
    assert config.require_reassessment_observation
    assert config.require_matching_observation_pairs
    assert config.allow_perturbation_observations
    assert config.allow_multiple_observations_per_key
    assert config.retain_last_plan
    assert config.retain_last_result


@pytest.mark.parametrize(
    "field_name",
    (
        "require_baseline_observation",
        "require_reassessment_observation",
        "require_matching_observation_pairs",
        "allow_perturbation_observations",
        "allow_multiple_observations_per_key",
        "retain_last_plan",
        "retain_last_result",
    ),
)
def test_engine_config_boolean_fields_require_bool(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngineConfig(**{field_name: 1})


def test_engine_config_validates_delta_mode() -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngineConfig(
            delta_mode="numeric",  # type: ignore[arg-type]
        )


# ============================================================================
# Construction and initial state
# ============================================================================


def test_engine_initial_state(
    registry: ProbeRegistry,
) -> None:
    engine = ActiveProbeEngine(registry)

    assert engine.state is ActiveProbeEngineState.IDLE
    assert engine.active_run is None
    assert engine.last_plan is None
    assert engine.last_result is None


def test_engine_accepts_registry_snapshot(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()
    engine = ActiveProbeEngine(snapshot)

    assert engine.registry is snapshot


def test_engine_rejects_invalid_registry() -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngine(object())  # type: ignore[arg-type]


def test_engine_rejects_invalid_policy(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngine(
            registry,
            policy=object(),  # type: ignore[arg-type]
        )


def test_engine_rejects_invalid_planner_config(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngine(
            registry,
            planner_config=object(),  # type: ignore[arg-type]
        )


def test_engine_rejects_invalid_config(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        ActiveProbeEngine(
            registry,
            config=object(),  # type: ignore[arg-type]
        )


# ============================================================================
# Planning lifecycle
# ============================================================================


def test_plan_changes_state_to_planned(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    result = engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )

    assert result.status is ProbeSelectionStatus.SELECTED
    assert engine.state is ActiveProbeEngineState.PLANNED
    assert engine.last_plan is result


def test_plan_uses_query_filter(
    engine: ActiveProbeEngine,
    passive_probe: ProbeDefinition,
) -> None:
    result = engine.plan(
        (
            make_estimate("probe.controlled"),
            make_estimate("probe.passive"),
        ),
        query=ProbeQuery(
            method=ProbeMethod.OBSERVATION,
        ),
    )

    assert result.probe_star is passive_probe


def test_plan_cannot_replace_running_probe(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(InvalidEngineStateError):
        engine.plan(
            (make_estimate(controlled_probe.identifier),)
        )


def test_last_plan_can_be_disabled(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            retain_last_plan=False,
        ),
    )

    engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )

    assert engine.last_plan is None
    assert engine.state is ActiveProbeEngineState.PLANNED


# ============================================================================
# Starting selected Probe
# ============================================================================


def test_start_selected_starts_probe_star(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    plan = engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )

    run = engine.start_selected(plan)

    assert run.definition is controlled_probe
    assert run.plan_candidate is plan.selected
    assert run.completion_status is None
    assert engine.state is ActiveProbeEngineState.RUNNING
    assert engine.active_run is run


def test_start_selected_uses_last_plan(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )

    run = engine.start_selected()

    assert run.definition is controlled_probe


def test_start_selected_requires_plan(
    engine: ActiveProbeEngine,
) -> None:
    with pytest.raises(InvalidEngineStateError):
        engine.start_selected()


def test_start_selected_rejects_nonselected_plan(
    engine: ActiveProbeEngine,
) -> None:
    plan = engine.plan(())

    with pytest.raises(InvalidEngineStateError):
        engine.start_selected(plan)


def test_start_selected_rejects_authorization_required_plan(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    policy = ProbePolicy(
        require_authorization_for_methods=frozenset(
            {ProbeMethod.INSTRUMENTAL}
        )
    )
    engine = ActiveProbeEngine(
        registry,
        policy=policy,
    )

    plan = engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )

    assert (
        plan.status
        is ProbeSelectionStatus.AUTHORIZATION_REQUIRED
    )

    with pytest.raises(ProbeAuthorizationRequiredError):
        engine.start_selected(plan)


def test_start_selected_rejects_wrong_plan_type(
    engine: ActiveProbeEngine,
) -> None:
    with pytest.raises(TypeError):
        engine.start_selected(object())  # type: ignore[arg-type]


# ============================================================================
# Direct start and Policy enforcement
# ============================================================================


def test_start_probe_by_identifier(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    run = engine.start_probe(controlled_probe.identifier)

    assert run.definition is controlled_probe
    assert run.plan_candidate is None
    assert engine.state is ActiveProbeEngineState.RUNNING


def test_start_probe_by_registered_definition(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    run = engine.start_probe(controlled_probe)

    assert run.definition is controlled_probe


def test_start_probe_rejects_unregistered_definition(
    engine: ActiveProbeEngine,
) -> None:
    foreign = make_probe("probe.foreign")

    with pytest.raises(Exception):
        engine.start_probe(foreign)


def test_start_probe_rejects_mismatched_definition(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    altered = ProbeDefinition(
        identifier=controlled_probe.identifier,
        name="Altered",
        method=controlled_probe.method,
        regime=controlled_probe.regime,
        purpose=controlled_probe.purpose,
        perturbation=controlled_probe.perturbation,
        tags=controlled_probe.tags,
    )

    with pytest.raises(ActiveProbeEngineError):
        engine.start_probe(altered)


def test_direct_start_cannot_bypass_policy_rejection(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(
            forbidden_methods=frozenset(
                {ProbeMethod.INSTRUMENTAL}
            )
        ),
    )

    with pytest.raises(ProbeExecutionRejectedError):
        engine.start_probe(controlled_probe.identifier)


def test_direct_start_cannot_bypass_authorization(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(
            require_authorization_for_methods=frozenset(
                {ProbeMethod.INSTRUMENTAL}
            )
        ),
    )

    with pytest.raises(ProbeAuthorizationRequiredError):
        engine.start_probe(controlled_probe.identifier)


def test_direct_start_accepts_authorized_context(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        policy=ProbePolicy(
            require_authorization_for_methods=frozenset(
                {ProbeMethod.INSTRUMENTAL}
            )
        ),
    )

    run = engine.start_probe(
        controlled_probe.identifier,
        context=ProbePolicyContext(
            human_authorized=True,
        ),
    )

    assert run.definition is controlled_probe


def test_direct_start_respects_non_fonit_veto(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    with pytest.raises(ProbeExecutionRejectedError):
        engine.start_probe(
            controlled_probe.identifier,
            assessment=ProbeAssessment(
                uncontrolled_propagation_possible=True,
            ),
            context=ProbePolicyContext(
                maximum_risk_level=ProbeRiskLevel.CRITICAL,
                maximum_cascade_risk=1.0,
                human_authorized=True,
            ),
        )


def test_cannot_start_second_probe_while_running(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
    passive_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(InvalidEngineStateError):
        engine.start_probe(passive_probe.identifier)


def test_run_ids_are_monotonic(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    first = engine.start_probe(controlled_probe.identifier)
    engine.cancel()
    engine.reset()

    second = engine.start_probe(controlled_probe.identifier)

    assert first.run_id == "probe-run-000001"
    assert second.run_id == "probe-run-000002"


# ============================================================================
# Observation collection
# ============================================================================


def test_add_observation_routes_by_phase(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    baseline = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )
    perturbation = make_observation(
        ProbePhase.PERTURBATION,
        value=4.0,
    )
    reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=1.5,
    )

    engine.add_observation(baseline)
    engine.add_observation(perturbation)
    engine.add_observation(reassessment)

    execution = engine.active_run.execution  # type: ignore[union-attr]

    assert execution.baseline_observations == [baseline]
    assert execution.perturbation_observations == [perturbation]
    assert execution.reassessment_observations == [reassessment]


def test_add_observation_requires_running_probe(
    engine: ActiveProbeEngine,
) -> None:
    with pytest.raises(ProbeExecutionNotFoundError):
        engine.add_observation(
            make_observation(
                ProbePhase.BASELINE,
                value=1.0,
            )
        )


def test_add_observation_validates_type(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(TypeError):
        engine.add_observation(object())  # type: ignore[arg-type]


def test_perturbation_observations_can_be_disabled(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            allow_perturbation_observations=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(ProbeObservationError):
        engine.add_observation(
            make_observation(
                ProbePhase.PERTURBATION,
                value=2.0,
            )
        )


def test_duplicate_key_can_be_rejected_within_phase(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            allow_multiple_observations_per_key=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )

    with pytest.raises(ProbeObservationError):
        engine.add_observation(
            make_observation(
                ProbePhase.BASELINE,
                value=2.0,
            )
        )


def test_same_key_is_allowed_across_different_phases(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            allow_multiple_observations_per_key=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=2.0,
        )
    )

    assert len(
        engine.active_run.execution.baseline_observations  # type: ignore[union-attr]
    ) == 1
    assert len(
        engine.active_run.execution.reassessment_observations  # type: ignore[union-attr]
    ) == 1


def test_add_observations_is_atomic_on_invalid_type(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    valid = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )

    with pytest.raises(TypeError):
        engine.add_observations(
            [valid, object()]  # type: ignore[list-item]
        )

    assert (
        engine.active_run.execution.baseline_observations  # type: ignore[union-attr]
        == []
    )


def test_add_observations_routes_all_phases(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    observations = [
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        ),
        make_observation(
            ProbePhase.PERTURBATION,
            value=3.0,
        ),
        make_observation(
            ProbePhase.REASSESSMENT,
            value=1.2,
        ),
    ]

    engine.add_observations(observations)

    execution = engine.active_run.execution  # type: ignore[union-attr]

    assert len(execution.baseline_observations) == 1
    assert len(execution.perturbation_observations) == 1
    assert len(execution.reassessment_observations) == 1


# ============================================================================
# Completion and derived deltas
# ============================================================================


def test_complete_creates_numeric_delta(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(
        engine,
        baseline=1.0,
        reassessment=1.5,
    )

    result = engine.complete()

    assert engine.state is ActiveProbeEngineState.COMPLETED
    assert len(result.deltas) == 1
    assert result.deltas[0].delta == pytest.approx(0.5)
    assert engine.last_result is result


def test_complete_creates_categorical_transition(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value="stable",
            units=None,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value="unstable",
            units=None,
        )
    )

    result = engine.complete()
    delta = result.deltas[0].delta

    assert isinstance(delta, ObservationTransition)
    assert delta.changed


def test_complete_requires_baseline_by_default(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=1.0,
        )
    )

    with pytest.raises(ProbeObservationError):
        engine.complete()


def test_complete_requires_reassessment_by_default(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )

    with pytest.raises(ProbeObservationError):
        engine.complete()


def test_baseline_requirement_can_be_disabled(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            require_baseline_observation=False,
            require_matching_observation_pairs=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=1.0,
        )
    )

    result = engine.complete()

    assert result.deltas == []


def test_reassessment_requirement_can_be_disabled(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            require_reassessment_observation=False,
            require_matching_observation_pairs=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)
    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )

    result = engine.complete()

    assert result.deltas == []


def test_mismatched_observation_keys_are_rejected(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
            target="node_A",
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=2.0,
            target="node_B",
        )
    )

    with pytest.raises(ProbeObservationError):
        engine.complete()


def test_mismatched_pair_counts_are_rejected(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.1,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=2.0,
        )
    )

    with pytest.raises(ProbeObservationError):
        engine.complete()


def test_multiple_pairs_preserve_insertion_order(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=1.0,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            value=2.0,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=1.5,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            value=3.0,
        )
    )

    result = engine.complete()

    assert [delta.delta for delta in result.deltas] == [
        pytest.approx(0.5),
        pytest.approx(1.0),
    ]


def test_custom_delta_calculator(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    result = engine.complete(
        delta_calculator=lambda before, after: {
            "before": before.value,
            "after": after.value,
        }
    )

    assert result.deltas[0].delta == {
        "before": 1.0,
        "after": 1.5,
    }


def test_delta_calculator_must_be_callable(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    with pytest.raises(TypeError):
        engine.complete(
            delta_calculator=object(),  # type: ignore[arg-type]
        )


def test_complete_validates_notes(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    with pytest.raises(TypeError):
        engine.complete(notes=123)  # type: ignore[arg-type]


def test_last_result_can_be_disabled(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            retain_last_result=False,
        ),
    )
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    engine.complete()

    assert engine.last_result is None


# ============================================================================
# Explicit deltas and GraphUpdate
# ============================================================================


def test_complete_accepts_explicit_delta(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    baseline, reassessment = add_numeric_pair(engine)

    explicit = ObservationDelta(
        baseline=baseline,
        reassessment=reassessment,
        delta={"custom": 42},
    )

    result = engine.complete(
        explicit_deltas=[explicit],
    )

    assert result.deltas == [explicit]


def test_explicit_delta_must_belong_to_active_execution(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    foreign_baseline = make_observation(
        ProbePhase.BASELINE,
        value=1.0,
    )
    foreign_reassessment = make_observation(
        ProbePhase.REASSESSMENT,
        value=2.0,
    )
    foreign_delta = ObservationDelta(
        baseline=foreign_baseline,
        reassessment=foreign_reassessment,
        delta=1.0,
    )

    with pytest.raises(ProbeObservationError):
        engine.complete(
            explicit_deltas=[foreign_delta],
        )


def test_explicit_deltas_validate_member_type(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    with pytest.raises(TypeError):
        engine.complete(
            explicit_deltas=[object()],  # type: ignore[list-item]
        )


def test_complete_attaches_graph_update_without_applying_it(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    baseline, reassessment = add_numeric_pair(engine)

    delta = ObservationDelta(
        baseline=baseline,
        reassessment=reassessment,
        delta=0.5,
    )
    graph_update = GraphUpdate(
        affected_nodes=("node_A",),
        affected_edges=(("node_A", "node_B"),),
        evidence=(delta,),
        confidence=0.9,
    )

    result = engine.complete(
        explicit_deltas=[delta],
        graph_update=graph_update,
    )

    assert result.graph_update is graph_update
    assert not hasattr(engine, "apply_graph_update")
    assert not hasattr(engine, "update_graph")


def test_complete_validates_graph_update_type(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)

    with pytest.raises(TypeError):
        engine.complete(
            graph_update=object(),  # type: ignore[arg-type]
        )


# ============================================================================
# Cancellation and reset
# ============================================================================


def test_cancel_running_probe(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    run = engine.cancel(reason="operator request")

    assert engine.state is ActiveProbeEngineState.CANCELLED
    assert (
        run.completion_status
        is ProbeCompletionStatus.CANCELLED
    )
    assert run.result is None
    assert (
        run.metadata["cancellation_reason"]
        == "operator request"
    )


def test_cancel_does_not_create_last_result(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.cancel()

    assert engine.last_result is None


def test_cancel_requires_running_probe(
    engine: ActiveProbeEngine,
) -> None:
    with pytest.raises(ProbeExecutionNotFoundError):
        engine.cancel()


def test_cancel_requires_nonempty_reason(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(ValueError):
        engine.cancel(reason=" ")


def test_reset_returns_completed_engine_to_idle(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)
    result = engine.complete()

    engine.reset()

    assert engine.state is ActiveProbeEngineState.IDLE
    assert engine.active_run is None
    assert engine.last_result is result


def test_reset_returns_cancelled_engine_to_idle(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.cancel()

    engine.reset()

    assert engine.state is ActiveProbeEngineState.IDLE
    assert engine.active_run is None


def test_reset_rejects_running_probe(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)

    with pytest.raises(InvalidEngineStateError):
        engine.reset()


def test_reset_can_clear_nonretained_history(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    engine = ActiveProbeEngine(
        registry,
        config=ActiveProbeEngineConfig(
            retain_last_plan=False,
            retain_last_result=False,
        ),
    )

    plan = engine.plan(
        (make_estimate(controlled_probe.identifier),)
    )
    engine.start_selected(plan)
    add_numeric_pair(engine)
    engine.complete()
    engine.reset()

    assert engine.last_plan is None
    assert engine.last_result is None


# ============================================================================
# Snapshot and counters
# ============================================================================


def test_initial_snapshot(
    engine: ActiveProbeEngine,
) -> None:
    snapshot = engine.snapshot()

    assert isinstance(snapshot, ActiveProbeEngineSnapshot)
    assert snapshot.state is ActiveProbeEngineState.IDLE
    assert snapshot.active_run is None
    assert snapshot.completed_run_count == 0
    assert snapshot.cancelled_run_count == 0


def test_snapshot_tracks_completed_count(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)
    result = engine.complete()

    snapshot = engine.snapshot()

    assert snapshot.state is ActiveProbeEngineState.COMPLETED
    assert snapshot.completed_run_count == 1
    assert snapshot.cancelled_run_count == 0
    assert snapshot.last_result is result


def test_snapshot_tracks_cancelled_count(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.cancel()

    snapshot = engine.snapshot()

    assert snapshot.cancelled_run_count == 1
    assert snapshot.completed_run_count == 0


def test_counters_survive_reset(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(controlled_probe.identifier)
    engine.cancel()
    engine.reset()

    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)
    engine.complete()
    engine.reset()

    snapshot = engine.snapshot()

    assert snapshot.completed_run_count == 1
    assert snapshot.cancelled_run_count == 1


def test_run_metadata_is_immutable(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    run = engine.start_probe(
        controlled_probe.identifier,
        metadata={"operator": "test"},
    )

    assert isinstance(run.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        run.metadata["operator"] = "changed"  # type: ignore[index]


def test_completion_merges_metadata(
    engine: ActiveProbeEngine,
    controlled_probe: ProbeDefinition,
) -> None:
    engine.start_probe(
        controlled_probe.identifier,
        metadata={"operator": "test"},
    )
    add_numeric_pair(engine)

    engine.complete(
        metadata={"session": "one"},
    )

    run = engine.active_run

    assert run is not None
    assert run.metadata["operator"] == "test"
    assert run.metadata["session"] == "one"


# ============================================================================
# ActiveProbeRun contracts
# ============================================================================


def test_completed_run_requires_result(
    controlled_probe: ProbeDefinition,
) -> None:
    execution = ProbeExecution(controlled_probe)
    policy_result = ProbePolicy().evaluate(controlled_probe)

    with pytest.raises(ValueError):
        ActiveProbeRun(
            run_id="run-1",
            definition=controlled_probe,
            execution=execution,
            policy_result=policy_result,
            completion_status=ProbeCompletionStatus.COMPLETED,
            result=None,
        )


def test_cancelled_run_rejects_result(
    controlled_probe: ProbeDefinition,
) -> None:
    execution = ProbeExecution(controlled_probe)
    policy_result = ProbePolicy().evaluate(controlled_probe)
    result = ProbeResult(
        execution=execution,
    )

    with pytest.raises(ValueError):
        ActiveProbeRun(
            run_id="run-1",
            definition=controlled_probe,
            execution=execution,
            policy_result=policy_result,
            completion_status=ProbeCompletionStatus.CANCELLED,
            result=result,
        )


def test_run_definition_must_match_execution(
    controlled_probe: ProbeDefinition,
    passive_probe: ProbeDefinition,
) -> None:
    execution = ProbeExecution(passive_probe)
    policy_result = ProbePolicy().evaluate(controlled_probe)

    with pytest.raises(ValueError):
        ActiveProbeRun(
            run_id="run-1",
            definition=controlled_probe,
            execution=execution,
            policy_result=policy_result,
        )


# ============================================================================
# Determinism and architectural boundaries
# ============================================================================


def test_identical_observations_produce_identical_deltas(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    first = ActiveProbeEngine(registry)
    first.start_probe(controlled_probe.identifier)
    add_numeric_pair(first)
    first_result = first.complete()

    second = ActiveProbeEngine(registry)
    second.start_probe(controlled_probe.identifier)
    add_numeric_pair(second)
    second_result = second.complete()

    assert (
        first_result.deltas[0].delta
        == second_result.deltas[0].delta
    )


def test_engine_does_not_mutate_registry(
    registry: ProbeRegistry,
    controlled_probe: ProbeDefinition,
) -> None:
    before = registry.definitions()

    engine = ActiveProbeEngine(registry)
    engine.start_probe(controlled_probe.identifier)
    add_numeric_pair(engine)
    engine.complete()

    assert registry.definitions() == before


def test_engine_has_no_solver_or_role_detection_methods(
    engine: ActiveProbeEngine,
) -> None:
    forbidden_methods = (
        "solve",
        "run_solver",
        "calculate_d_origin",
        "calculate_d_fast",
        "calculate_d_root",
        "calculate_node_star",
        "apply_graph_update",
        "mutate_graph",
    )

    for method_name in forbidden_methods:
        assert not hasattr(engine, method_name)


def test_engine_does_not_physically_execute_perturbation(
    engine: ActiveProbeEngine,
) -> None:
    forbidden_methods = (
        "apply_load",
        "apply_force",
        "apply_perturbation",
        "perform_perturbation",
    )

    for method_name in forbidden_methods:
        assert not hasattr(engine, method_name)
