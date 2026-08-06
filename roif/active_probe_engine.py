"""
ROIF Engine
===========

Universal orchestration layer for the Active Probe Engine.

The ActiveProbeEngine connects:

    ProbeRegistry
        ->
    ProbePolicy
        ->
    ProbePlanner
        ->
    Probe*
        ->
    ProbeExecution
        ->
    ProbeResult

Architectural boundaries
------------------------

The engine:

- does not physically execute perturbations;
- does not directly mutate the investigated graph;
- does not call the main ROIF Solver;
- does not calculate D_origin, D_fast, D_root, or Node*;
- does not contain medical or other domain-specific logic.

It manages the lifecycle of a controlled experiment and produces an auditable
ProbeResult that may later be consumed by a graph-update layer or Solver.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any

from .probe_entities import (
    GraphUpdate,
    Observation,
    ObservationDelta,
    ProbeDefinition,
    ProbeExecution,
    ProbePhase,
    ProbeResult,
)
from .probe_planner import (
    ProbeInformationEstimate,
    ProbePlanCandidate,
    ProbePlanner,
    ProbePlannerConfig,
    ProbePlanResult,
    ProbeSelectionStatus,
)
from .probe_policy import (
    ProbeAssessment,
    ProbePolicy,
    ProbePolicyContext,
    ProbePolicyDecision,
    ProbePolicyResult,
)
from .probe_registry import (
    ProbeQuery,
    ProbeRegistry,
    ProbeRegistrySnapshot,
)


# ============================================================================
# Errors
# ============================================================================


class ActiveProbeEngineError(Exception):
    """Base exception for Active Probe Engine failures."""


class InvalidEngineStateError(ActiveProbeEngineError):
    """Raised when an operation is invalid in the current engine state."""


class ProbeAuthorizationRequiredError(ActiveProbeEngineError):
    """Raised when Probe execution requires human authorization."""


class ProbeExecutionRejectedError(ActiveProbeEngineError):
    """Raised when Probe Policy rejects direct execution."""


class ProbeObservationError(ActiveProbeEngineError):
    """Raised when observations cannot form a valid Probe result."""


class ProbeExecutionNotFoundError(ActiveProbeEngineError):
    """Raised when no active Probe execution exists."""


# ============================================================================
# Engine taxonomy
# ============================================================================


class ActiveProbeEngineState(str, Enum):
    """Current lifecycle state of ActiveProbeEngine."""

    IDLE = "idle"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ProbeCompletionStatus(str, Enum):
    """Outcome assigned to a finished Probe execution."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ObservationDeltaMode(str, Enum):
    """Default strategy used to derive ObservationDelta values."""

    NUMERIC_OR_TRANSITION = "numeric_or_transition"
    NUMERIC_ONLY = "numeric_only"
    EXPLICIT_ONLY = "explicit_only"


# ============================================================================
# Validation helpers
# ============================================================================


def _freeze_mapping(
    mapping: Mapping[str, Any],
    field_name: str = "metadata",
) -> Mapping[str, Any]:
    """Return an immutable shallow copy of a mapping."""

    if not isinstance(mapping, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    return MappingProxyType(dict(mapping))


def _validate_nonempty_text(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized


def _validate_positive_int(
    value: int,
    field_name: str,
) -> int:
    """Validate a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")

    return value


# ============================================================================
# Observation transitions
# ============================================================================


@dataclass(frozen=True, slots=True)
class ObservationTransition:
    """
    Non-numeric transition between baseline and reassessment values.

    This preserves the original values without imposing domain-specific
    arithmetic on categorical, logical, vector, or structured observations.
    """

    before: Any
    after: Any

    @property
    def changed(self) -> bool:
        """Return whether the two values differ."""

        try:
            equality = self.before == self.after
        except Exception:
            return True

        if isinstance(equality, bool):
            return not equality

        return True


ObservationDeltaCalculator = Callable[
    [Observation, Observation],
    Any,
]


def default_observation_delta(
    baseline: Observation,
    reassessment: Observation,
    *,
    mode: ObservationDeltaMode = (
        ObservationDeltaMode.NUMERIC_OR_TRANSITION
    ),
) -> Any:
    """
    Derive a domain-independent delta value.

    Numeric observations produce:

        reassessment.value - baseline.value

    Non-numeric observations produce ObservationTransition unless the selected
    mode requires numeric or explicitly supplied deltas.
    """

    if not isinstance(baseline, Observation):
        raise TypeError("baseline must be Observation")

    if not isinstance(reassessment, Observation):
        raise TypeError("reassessment must be Observation")

    if not isinstance(mode, ObservationDeltaMode):
        raise TypeError("mode must be ObservationDeltaMode")

    baseline_value = baseline.value
    reassessment_value = reassessment.value

    numeric = (
        isinstance(baseline_value, Real)
        and not isinstance(baseline_value, bool)
        and isinstance(reassessment_value, Real)
        and not isinstance(reassessment_value, bool)
    )

    if numeric:
        before = float(baseline_value)
        after = float(reassessment_value)

        if not math.isfinite(before):
            raise ProbeObservationError(
                "baseline numeric value must be finite"
            )

        if not math.isfinite(after):
            raise ProbeObservationError(
                "reassessment numeric value must be finite"
            )

        return after - before

    if mode is ObservationDeltaMode.NUMERIC_ONLY:
        raise ProbeObservationError(
            "non-numeric observations require an explicit delta calculator"
        )

    if mode is ObservationDeltaMode.EXPLICIT_ONLY:
        raise ProbeObservationError(
            "automatic delta calculation is disabled"
        )

    return ObservationTransition(
        before=baseline_value,
        after=reassessment_value,
    )


# ============================================================================
# Engine configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ActiveProbeEngineConfig:
    """Configuration controlling Probe execution lifecycle."""

    require_baseline_observation: bool = True
    require_reassessment_observation: bool = True
    require_matching_observation_pairs: bool = True

    allow_perturbation_observations: bool = True
    allow_multiple_observations_per_key: bool = True

    delta_mode: ObservationDeltaMode = (
        ObservationDeltaMode.NUMERIC_OR_TRANSITION
    )

    retain_last_plan: bool = True
    retain_last_result: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "require_baseline_observation",
            "require_reassessment_observation",
            "require_matching_observation_pairs",
            "allow_perturbation_observations",
            "allow_multiple_observations_per_key",
            "retain_last_plan",
            "retain_last_result",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

        if not isinstance(self.delta_mode, ObservationDeltaMode):
            raise TypeError(
                "delta_mode must be ObservationDeltaMode"
            )


# ============================================================================
# Execution records
# ============================================================================


@dataclass(frozen=True, slots=True)
class ActiveProbeRun:
    """Immutable description of one active or completed Probe run."""

    run_id: str
    definition: ProbeDefinition
    execution: ProbeExecution

    policy_result: ProbePolicyResult
    plan_candidate: ProbePlanCandidate | None = None

    completion_status: ProbeCompletionStatus | None = None
    result: ProbeResult | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _validate_nonempty_text(self.run_id, "run_id"),
        )

        if not isinstance(self.definition, ProbeDefinition):
            raise TypeError(
                "definition must be ProbeDefinition"
            )

        if not isinstance(self.execution, ProbeExecution):
            raise TypeError(
                "execution must be ProbeExecution"
            )

        if self.execution.definition is not self.definition:
            raise ValueError(
                "execution definition must match run definition"
            )

        if not isinstance(self.policy_result, ProbePolicyResult):
            raise TypeError(
                "policy_result must be ProbePolicyResult"
            )

        if (
            self.plan_candidate is not None
            and not isinstance(
                self.plan_candidate,
                ProbePlanCandidate,
            )
        ):
            raise TypeError(
                "plan_candidate must be ProbePlanCandidate or None"
            )

        if (
            self.completion_status is not None
            and not isinstance(
                self.completion_status,
                ProbeCompletionStatus,
            )
        ):
            raise TypeError(
                "completion_status must be "
                "ProbeCompletionStatus or None"
            )

        if (
            self.result is not None
            and not isinstance(self.result, ProbeResult)
        ):
            raise TypeError(
                "result must be ProbeResult or None"
            )

        if (
            self.completion_status
            is ProbeCompletionStatus.COMPLETED
            and self.result is None
        ):
            raise ValueError(
                "completed run must contain ProbeResult"
            )

        if (
            self.completion_status
            is ProbeCompletionStatus.CANCELLED
            and self.result is not None
        ):
            raise ValueError(
                "cancelled run must not contain ProbeResult"
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class ActiveProbeEngineSnapshot:
    """Immutable point-in-time view of ActiveProbeEngine."""

    state: ActiveProbeEngineState
    active_run: ActiveProbeRun | None

    last_plan: ProbePlanResult | None
    last_result: ProbeResult | None

    completed_run_count: int
    cancelled_run_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, ActiveProbeEngineState):
            raise TypeError(
                "state must be ActiveProbeEngineState"
            )

        if (
            self.active_run is not None
            and not isinstance(self.active_run, ActiveProbeRun)
        ):
            raise TypeError(
                "active_run must be ActiveProbeRun or None"
            )

        if (
            self.last_plan is not None
            and not isinstance(self.last_plan, ProbePlanResult)
        ):
            raise TypeError(
                "last_plan must be ProbePlanResult or None"
            )

        if (
            self.last_result is not None
            and not isinstance(self.last_result, ProbeResult)
        ):
            raise TypeError(
                "last_result must be ProbeResult or None"
            )

        _validate_positive_int(
            self.completed_run_count + 1,
            "completed_run_count + 1",
        )
        _validate_positive_int(
            self.cancelled_run_count + 1,
            "cancelled_run_count + 1",
        )


# ============================================================================
# Active Probe Engine
# ============================================================================


class ActiveProbeEngine:
    """
    Universal lifecycle coordinator for Active Probe experiments.

    Lifecycle
    ---------

    IDLE
        ->
    PLANNED
        ->
    RUNNING
        ->
    COMPLETED

    A running Probe may also transition to CANCELLED.
    """

    __slots__ = (
        "_registry",
        "_policy",
        "_planner",
        "_config",
        "_state",
        "_active_run",
        "_last_plan",
        "_last_result",
        "_run_counter",
        "_completed_run_count",
        "_cancelled_run_count",
    )

    def __init__(
        self,
        registry: ProbeRegistry | ProbeRegistrySnapshot,
        *,
        policy: ProbePolicy | None = None,
        planner_config: ProbePlannerConfig | None = None,
        config: ActiveProbeEngineConfig | None = None,
    ) -> None:
        if not isinstance(
            registry,
            (ProbeRegistry, ProbeRegistrySnapshot),
        ):
            raise TypeError(
                "registry must be ProbeRegistry "
                "or ProbeRegistrySnapshot"
            )

        if policy is None:
            policy = ProbePolicy()
        elif not isinstance(policy, ProbePolicy):
            raise TypeError(
                "policy must be ProbePolicy or None"
            )

        if planner_config is None:
            planner_config = ProbePlannerConfig()
        elif not isinstance(
            planner_config,
            ProbePlannerConfig,
        ):
            raise TypeError(
                "planner_config must be ProbePlannerConfig or None"
            )

        if config is None:
            config = ActiveProbeEngineConfig()
        elif not isinstance(config, ActiveProbeEngineConfig):
            raise TypeError(
                "config must be ActiveProbeEngineConfig or None"
            )

        self._registry = registry
        self._policy = policy
        self._planner = ProbePlanner(
            registry=registry,
            policy=policy,
            config=planner_config,
        )
        self._config = config

        self._state = ActiveProbeEngineState.IDLE
        self._active_run: ActiveProbeRun | None = None
        self._last_plan: ProbePlanResult | None = None
        self._last_result: ProbeResult | None = None

        self._run_counter = 0
        self._completed_run_count = 0
        self._cancelled_run_count = 0

    @property
    def registry(
        self,
    ) -> ProbeRegistry | ProbeRegistrySnapshot:
        return self._registry

    @property
    def policy(self) -> ProbePolicy:
        return self._policy

    @property
    def planner(self) -> ProbePlanner:
        return self._planner

    @property
    def config(self) -> ActiveProbeEngineConfig:
        return self._config

    @property
    def state(self) -> ActiveProbeEngineState:
        return self._state

    @property
    def active_run(self) -> ActiveProbeRun | None:
        return self._active_run

    @property
    def last_plan(self) -> ProbePlanResult | None:
        return self._last_plan

    @property
    def last_result(self) -> ProbeResult | None:
        return self._last_result

    def plan(
        self,
        estimates: (
            Mapping[str, ProbeInformationEstimate]
            | tuple[ProbeInformationEstimate, ...]
            | list[ProbeInformationEstimate]
        ),
        *,
        assessments: Mapping[str, ProbeAssessment] | None = None,
        context: ProbePolicyContext | None = None,
        query: ProbeQuery | None = None,
    ) -> ProbePlanResult:
        """Plan and rank the next Probe candidates."""

        self._ensure_not_running("plan")

        result = self._planner.plan(
            estimates,
            assessments=assessments,
            context=context,
            query=query,
        )

        if self._config.retain_last_plan:
            self._last_plan = result

        self._state = ActiveProbeEngineState.PLANNED

        return result

    def start_selected(
        self,
        plan_result: ProbePlanResult | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActiveProbeRun:
        """Start the Probe* selected by a successful planning result."""

        self._ensure_not_running("start_selected")

        if plan_result is None:
            plan_result = self._last_plan

        if plan_result is None:
            raise InvalidEngineStateError(
                "no Probe plan is available"
            )

        if not isinstance(plan_result, ProbePlanResult):
            raise TypeError(
                "plan_result must be ProbePlanResult or None"
            )

        if (
            plan_result.status
            is ProbeSelectionStatus.AUTHORIZATION_REQUIRED
        ):
            raise ProbeAuthorizationRequiredError(
                "selected Probe requires human authorization"
            )

        if (
            plan_result.status
            is not ProbeSelectionStatus.SELECTED
            or plan_result.selected is None
        ):
            raise InvalidEngineStateError(
                "plan does not contain an executable Probe*"
            )

        candidate = plan_result.selected

        return self._start_run(
            definition=candidate.definition,
            policy_result=candidate.policy_result,
            plan_candidate=candidate,
            metadata=metadata,
        )

    def start_probe(
        self,
        probe: str | ProbeDefinition,
        *,
        assessment: ProbeAssessment | None = None,
        context: ProbePolicyContext | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActiveProbeRun:
        """
        Start a Probe directly after mandatory policy evaluation.

        Direct execution bypasses ranking, but never bypasses ProbePolicy.
        """

        self._ensure_not_running("start_probe")

        definition = self._resolve_definition(probe)

        if assessment is None:
            assessment = ProbeAssessment()
        elif not isinstance(assessment, ProbeAssessment):
            raise TypeError(
                "assessment must be ProbeAssessment or None"
            )

        if context is None:
            context = ProbePolicyContext()
        elif not isinstance(context, ProbePolicyContext):
            raise TypeError(
                "context must be ProbePolicyContext or None"
            )

        policy_result = self._policy.evaluate(
            definition,
            assessment,
            context,
        )

        if (
            policy_result.decision
            is ProbePolicyDecision.REJECT
        ):
            raise ProbeExecutionRejectedError(
                f"Probe rejected by policy: "
                f"{definition.identifier!r}"
            )

        if (
            policy_result.decision
            is ProbePolicyDecision.REQUIRE_AUTHORIZATION
        ):
            raise ProbeAuthorizationRequiredError(
                f"Probe requires authorization: "
                f"{definition.identifier!r}"
            )

        return self._start_run(
            definition=definition,
            policy_result=policy_result,
            plan_candidate=None,
            metadata=metadata,
        )

    def add_observation(
        self,
        observation: Observation,
    ) -> None:
        """Add one observation to the active Probe execution."""

        run = self._require_active_run()

        if not isinstance(observation, Observation):
            raise TypeError(
                "observation must be Observation"
            )

        if (
            observation.phase is ProbePhase.PERTURBATION
            and not self._config.allow_perturbation_observations
        ):
            raise ProbeObservationError(
                "perturbation observations are disabled"
            )

        if not self._config.allow_multiple_observations_per_key:
            collection = self._collection_for_phase(
                run.execution,
                observation.phase,
            )

            key = self._observation_key(observation)

            if any(
                self._observation_key(existing) == key
                for existing in collection
            ):
                raise ProbeObservationError(
                    "duplicate observation key is not allowed: "
                    f"{key!r}"
                )

        run.execution.add_observation(observation)

    def add_observations(
        self,
        observations: tuple[Observation, ...] | list[Observation],
    ) -> None:
        """Add multiple observations atomically after validation."""

        run = self._require_active_run()

        if isinstance(observations, (str, bytes)):
            raise TypeError(
                "observations must be a sequence of Observation objects"
            )

        incoming = tuple(observations)

        for observation in incoming:
            if not isinstance(observation, Observation):
                raise TypeError(
                    "observations must contain only Observation instances"
                )

            if (
                observation.phase is ProbePhase.PERTURBATION
                and not self._config.allow_perturbation_observations
            ):
                raise ProbeObservationError(
                    "perturbation observations are disabled"
                )

        if not self._config.allow_multiple_observations_per_key:
            existing_keys = {
                self._observation_key(observation)
                for observation in (
                    *run.execution.baseline_observations,
                    *run.execution.perturbation_observations,
                    *run.execution.reassessment_observations,
                )
            }

            incoming_keys: set[
                tuple[Any, ...]
            ] = set()

            for observation in incoming:
                key = self._observation_key(observation)

                if key in existing_keys or key in incoming_keys:
                    raise ProbeObservationError(
                        "duplicate observation key is not allowed: "
                        f"{key!r}"
                    )

                incoming_keys.add(key)

        for observation in incoming:
            run.execution.add_observation(observation)

    def complete(
        self,
        *,
        explicit_deltas: (
            tuple[ObservationDelta, ...]
            | list[ObservationDelta]
            | None
        ) = None,
        delta_calculator: ObservationDeltaCalculator | None = None,
        graph_update: GraphUpdate | None = None,
        notes: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ProbeResult:
        """Complete the active Probe and create a ProbeResult."""

        run = self._require_active_run()
        execution = run.execution

        self._validate_completion_observations(execution)

        if explicit_deltas is not None:
            deltas = self._validate_explicit_deltas(
                explicit_deltas,
                execution,
            )
        else:
            deltas = self._derive_deltas(
                execution,
                delta_calculator=delta_calculator,
            )

        if graph_update is not None and not isinstance(
            graph_update,
            GraphUpdate,
        ):
            raise TypeError(
                "graph_update must be GraphUpdate or None"
            )

        if not isinstance(notes, str):
            raise TypeError("notes must be a string")

        result = ProbeResult(
            execution=execution,
            deltas=list(deltas),
            graph_update=graph_update,
            notes=notes,
        )

        completed_metadata = dict(run.metadata)

        if metadata is not None:
            completed_metadata.update(
                dict(_freeze_mapping(metadata))
            )

        completed_run = ActiveProbeRun(
            run_id=run.run_id,
            definition=run.definition,
            execution=execution,
            policy_result=run.policy_result,
            plan_candidate=run.plan_candidate,
            completion_status=ProbeCompletionStatus.COMPLETED,
            result=result,
            metadata=completed_metadata,
        )

        self._active_run = completed_run
        self._completed_run_count += 1
        self._state = ActiveProbeEngineState.COMPLETED

        if self._config.retain_last_result:
            self._last_result = result

        return result

    def cancel(
        self,
        *,
        reason: str = "cancelled",
        metadata: Mapping[str, Any] | None = None,
    ) -> ActiveProbeRun:
        """Cancel the active Probe without producing ProbeResult."""

        run = self._require_active_run()

        normalized_reason = _validate_nonempty_text(
            reason,
            "reason",
        )

        cancelled_metadata = dict(run.metadata)
        cancelled_metadata["cancellation_reason"] = normalized_reason

        if metadata is not None:
            cancelled_metadata.update(
                dict(_freeze_mapping(metadata))
            )

        cancelled_run = ActiveProbeRun(
            run_id=run.run_id,
            definition=run.definition,
            execution=run.execution,
            policy_result=run.policy_result,
            plan_candidate=run.plan_candidate,
            completion_status=ProbeCompletionStatus.CANCELLED,
            result=None,
            metadata=cancelled_metadata,
        )

        self._active_run = cancelled_run
        self._cancelled_run_count += 1
        self._state = ActiveProbeEngineState.CANCELLED

        return cancelled_run

    def reset(self) -> None:
        """
        Reset transient lifecycle state.

        Registered Probe definitions, policy, planner configuration, and run
        counters remain unchanged.
        """

        if self._state is ActiveProbeEngineState.RUNNING:
            raise InvalidEngineStateError(
                "running Probe must be completed or cancelled before reset"
            )

        self._active_run = None
        self._state = ActiveProbeEngineState.IDLE

        if not self._config.retain_last_plan:
            self._last_plan = None

        if not self._config.retain_last_result:
            self._last_result = None

    def snapshot(self) -> ActiveProbeEngineSnapshot:
        """Return an immutable point-in-time engine snapshot."""

        return ActiveProbeEngineSnapshot(
            state=self._state,
            active_run=self._active_run,
            last_plan=self._last_plan,
            last_result=self._last_result,
            completed_run_count=self._completed_run_count,
            cancelled_run_count=self._cancelled_run_count,
        )

    def _start_run(
        self,
        *,
        definition: ProbeDefinition,
        policy_result: ProbePolicyResult,
        plan_candidate: ProbePlanCandidate | None,
        metadata: Mapping[str, Any] | None,
    ) -> ActiveProbeRun:
        """Create and activate one Probe run."""

        self._run_counter += 1

        run_id = f"probe-run-{self._run_counter:06d}"

        execution = ProbeExecution(
            definition=definition,
        )

        run = ActiveProbeRun(
            run_id=run_id,
            definition=definition,
            execution=execution,
            policy_result=policy_result,
            plan_candidate=plan_candidate,
            metadata=(
                {}
                if metadata is None
                else _freeze_mapping(metadata)
            ),
        )

        self._active_run = run
        self._state = ActiveProbeEngineState.RUNNING

        return run

    def _resolve_definition(
        self,
        probe: str | ProbeDefinition,
    ) -> ProbeDefinition:
        """Resolve a ProbeDefinition from identifier or instance."""

        if isinstance(probe, ProbeDefinition):
            registered = self._get_registered_definition(
                probe.identifier
            )

            if registered != probe:
                raise ActiveProbeEngineError(
                    "ProbeDefinition does not match registered definition"
                )

            return registered

        if isinstance(probe, str):
            return self._get_registered_definition(probe)

        raise TypeError(
            "probe must be a Probe identifier or ProbeDefinition"
        )

    def _get_registered_definition(
        self,
        identifier: str,
    ) -> ProbeDefinition:
        """Return one registered ProbeDefinition."""

        if isinstance(self._registry, ProbeRegistry):
            return self._registry.get(identifier)

        return self._registry.get(identifier)

    def _require_active_run(self) -> ActiveProbeRun:
        """Return active running Probe or raise."""

        if (
            self._state is not ActiveProbeEngineState.RUNNING
            or self._active_run is None
        ):
            raise ProbeExecutionNotFoundError(
                "no running Probe execution exists"
            )

        return self._active_run

    def _ensure_not_running(
        self,
        operation: str,
    ) -> None:
        """Reject operations that would replace a running Probe."""

        if self._state is ActiveProbeEngineState.RUNNING:
            raise InvalidEngineStateError(
                f"cannot {operation} while a Probe is running"
            )

    def _validate_completion_observations(
        self,
        execution: ProbeExecution,
    ) -> None:
        """Validate observations required to complete a Probe."""

        if (
            self._config.require_baseline_observation
            and not execution.baseline_observations
        ):
            raise ProbeObservationError(
                "at least one baseline observation is required"
            )

        if (
            self._config.require_reassessment_observation
            and not execution.reassessment_observations
        ):
            raise ProbeObservationError(
                "at least one reassessment observation is required"
            )

    def _derive_deltas(
        self,
        execution: ProbeExecution,
        *,
        delta_calculator: ObservationDeltaCalculator | None,
    ) -> tuple[ObservationDelta, ...]:
        """Pair baseline/reassessment observations and derive deltas."""

        if delta_calculator is not None and not callable(
            delta_calculator
        ):
            raise TypeError(
                "delta_calculator must be callable or None"
            )

        baseline_groups = self._group_observations(
            execution.baseline_observations
        )
        reassessment_groups = self._group_observations(
            execution.reassessment_observations
        )

        all_keys = tuple(
            sorted(
                set(baseline_groups)
                | set(reassessment_groups),
                key=repr,
            )
        )

        deltas: list[ObservationDelta] = []

        for key in all_keys:
            baseline_items = baseline_groups.get(key, ())
            reassessment_items = reassessment_groups.get(key, ())

            if len(baseline_items) != len(reassessment_items):
                if self._config.require_matching_observation_pairs:
                    raise ProbeObservationError(
                        "baseline and reassessment counts do not match "
                        f"for observation key {key!r}"
                    )

            pair_count = min(
                len(baseline_items),
                len(reassessment_items),
            )

            for index in range(pair_count):
                baseline = baseline_items[index]
                reassessment = reassessment_items[index]

                if delta_calculator is None:
                    delta_value = default_observation_delta(
                        baseline,
                        reassessment,
                        mode=self._config.delta_mode,
                    )
                else:
                    delta_value = delta_calculator(
                        baseline,
                        reassessment,
                    )

                deltas.append(
                    ObservationDelta(
                        baseline=baseline,
                        reassessment=reassessment,
                        delta=delta_value,
                    )
                )

        if (
            self._config.require_matching_observation_pairs
            and not deltas
            and (
                execution.baseline_observations
                or execution.reassessment_observations
            )
        ):
            raise ProbeObservationError(
                "no compatible baseline/reassessment pairs were found"
            )

        return tuple(deltas)

    @staticmethod
    def _validate_explicit_deltas(
        deltas: tuple[ObservationDelta, ...]
        | list[ObservationDelta],
        execution: ProbeExecution,
    ) -> tuple[ObservationDelta, ...]:
        """Validate caller-supplied ObservationDelta objects."""

        if isinstance(deltas, (str, bytes)):
            raise TypeError(
                "explicit_deltas must be a sequence "
                "of ObservationDelta objects"
            )

        normalized = tuple(deltas)

        baseline_ids = {
            id(observation)
            for observation in execution.baseline_observations
        }
        reassessment_ids = {
            id(observation)
            for observation
            in execution.reassessment_observations
        }

        for delta in normalized:
            if not isinstance(delta, ObservationDelta):
                raise TypeError(
                    "explicit_deltas must contain only "
                    "ObservationDelta instances"
                )

            if id(delta.baseline) not in baseline_ids:
                raise ProbeObservationError(
                    "explicit delta baseline does not belong "
                    "to active execution"
                )

            if id(delta.reassessment) not in reassessment_ids:
                raise ProbeObservationError(
                    "explicit delta reassessment does not belong "
                    "to active execution"
                )

        return normalized

    @staticmethod
    def _observation_key(
        observation: Observation,
    ) -> tuple[Any, ...]:
        """Return the pairing key for one observation."""

        return (
            observation.channel,
            observation.target,
            observation.units,
        )

    @classmethod
    def _group_observations(
        cls,
        observations: list[Observation],
    ) -> dict[tuple[Any, ...], tuple[Observation, ...]]:
        """Group observations by channel, target, and units."""

        mutable: dict[
            tuple[Any, ...],
            list[Observation],
        ] = {}

        for observation in observations:
            key = cls._observation_key(observation)
            mutable.setdefault(key, []).append(observation)

        return {
            key: tuple(values)
            for key, values in mutable.items()
        }

    @staticmethod
    def _collection_for_phase(
        execution: ProbeExecution,
        phase: ProbePhase,
    ) -> list[Observation]:
        """Return the execution collection matching a Probe phase."""

        if phase is ProbePhase.BASELINE:
            return execution.baseline_observations

        if phase is ProbePhase.PERTURBATION:
            return execution.perturbation_observations

        if phase is ProbePhase.REASSESSMENT:
            return execution.reassessment_observations

        raise ProbeObservationError(
            f"unsupported Probe phase: {phase!r}"
        )


__all__ = [
    "ActiveProbeEngine",
    "ActiveProbeEngineConfig",
    "ActiveProbeEngineError",
    "ActiveProbeEngineSnapshot",
    "ActiveProbeEngineState",
    "ActiveProbeRun",
    "InvalidEngineStateError",
    "ObservationDeltaCalculator",
    "ObservationDeltaMode",
    "ObservationTransition",
    "ProbeAuthorizationRequiredError",
    "ProbeCompletionStatus",
    "ProbeExecutionNotFoundError",
    "ProbeExecutionRejectedError",
    "ProbeObservationError",
    "default_observation_delta",
]
