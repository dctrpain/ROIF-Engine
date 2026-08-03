from __future__ import annotations

"""
Candidate-history generation for the ROIF Structural Memory Engine.

HistoryDecoder ranks already constructed candidate histories.

HypothesisGenerator answers an earlier question:

    Which candidate histories are scientifically admissible enough to test?

The generator does not declare a cause true and does not fabricate a
predicted StructuralSignature. Candidate hypotheses are produced from:

- an observed StructuralSignature;
- explicit causal rules;
- optional active influence planes;
- optional known agents;
- optional candidate roots;
- temporal and profile constraints;
- an external forward predictor.

The forward predictor remains responsible for producing the predicted
StructuralSignature of each candidate. This preserves the scientific
separation:

    rule-based candidate generation
        -> forward reconstruction
        -> predicted signature
        -> HistoryDecoder ranking

The implementation is deterministic, immutable, serializable, and
domain-independent.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from .history_decoder import (
    HistoryHypothesis,
    HypothesisStatus,
)
from .structural_signature import (
    StructuralSignature,
)


class HypothesisGeneratorError(ValueError):
    """Raised when hypothesis generation data are invalid."""


class RuleMatchMode(str, Enum):
    """How a causal rule compares its profile constraints."""

    ANY = "any"
    ALL = "all"


class CandidateStatus(str, Enum):
    """Internal generation status before decoder ranking."""

    GENERATED = "generated"
    PREDICTED = "predicted"
    SKIPPED = "skipped"
    FAILED = "failed"


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise HypothesisGeneratorError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise HypothesisGeneratorError(
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
        raise HypothesisGeneratorError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise HypothesisGeneratorError(
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
        raise HypothesisGeneratorError(
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
        raise HypothesisGeneratorError(
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
        raise HypothesisGeneratorError(
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


def _normalize_ids(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_value in values:
        value = _require_nonempty_string(
            raw_value,
            field_name=field_name,
        )

        if value in seen:
            continue

        seen.add(value)
        normalized.append(value)

    return tuple(normalized)


def _normalize_thresholds(
    value: Mapping[str, float] | None,
    *,
    field_name: str,
) -> Mapping[str, float]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise HypothesisGeneratorError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, float] = {}

    for raw_key, raw_threshold in value.items():
        key = _require_nonempty_string(
            raw_key,
            field_name=f"{field_name} key",
        )
        threshold = _validate_unit_interval(
            raw_threshold,
            field_name=f"{field_name}[{key!r}]",
        )
        normalized[key] = threshold

    return MappingProxyType(
        dict(sorted(normalized.items()))
    )


class SignaturePredictor(Protocol):
    """
    Forward-prediction contract used by HypothesisGenerator.

    Implementations may call:

    - the ROIF forward solver;
    - an analytical reconstruction;
    - an empirical model;
    - a validated template library.

    They must return a StructuralSignature.
    """

    def __call__(
        self,
        seed: HypothesisSeed,
        observed: StructuralSignature,
    ) -> StructuralSignature:
        ...


@dataclass(frozen=True, slots=True)
class CausalRule:
    """
    Explicit rule describing one scientifically admissible causal family.

    A rule can require evidence in one or more signature profiles.

    Example
    -------
    A mechanical-impact rule may require:

        plane_thresholds={"mechanical": 0.20}
        kind_thresholds={"damage": 0.10}
        allowed_time_scales=("instant", "fast", "medium")

    Rules generate candidates. They do not prove causality.
    """

    name: str
    hypothesis_name: str

    description: str | None = None

    initiating_plane_ids: tuple[str, ...] = ()
    initiating_agent_ids: tuple[str, ...] = ()
    candidate_root_ids: tuple[str, ...] = ()

    plane_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )
    agent_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )
    kind_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )
    time_scale_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )
    target_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )

    match_mode: RuleMatchMode = RuleMatchMode.ALL

    minimum_total_changes: int = 0
    minimum_causal_depth: int = 0

    minimum_capacity_loss: float = 0.0
    minimum_capacity_gain: float = 0.0
    minimum_persistence: float = 0.0
    minimum_irreversibility: float = 0.0

    prior_probability: float = 0.5
    validation_score: float = 0.5
    complexity: float = 1.0
    custom_penalty: float = 0.0

    enabled: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    rule_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        name = _require_nonempty_string(
            self.name,
            field_name="name",
        )
        hypothesis_name = _require_nonempty_string(
            self.hypothesis_name,
            field_name="hypothesis_name",
        )
        rule_id = _require_nonempty_string(
            self.rule_id,
            field_name="rule_id",
        )

        description = self.description

        if description is not None:
            description = _require_nonempty_string(
                description,
                field_name="description",
            )

        initiating_plane_ids = _normalize_ids(
            self.initiating_plane_ids,
            field_name="initiating_plane_ids",
        )
        initiating_agent_ids = _normalize_ids(
            self.initiating_agent_ids,
            field_name="initiating_agent_ids",
        )
        candidate_root_ids = _normalize_ids(
            self.candidate_root_ids,
            field_name="candidate_root_ids",
        )

        plane_thresholds = _normalize_thresholds(
            self.plane_thresholds,
            field_name="plane_thresholds",
        )
        agent_thresholds = _normalize_thresholds(
            self.agent_thresholds,
            field_name="agent_thresholds",
        )
        kind_thresholds = _normalize_thresholds(
            self.kind_thresholds,
            field_name="kind_thresholds",
        )
        time_scale_thresholds = _normalize_thresholds(
            self.time_scale_thresholds,
            field_name="time_scale_thresholds",
        )
        target_thresholds = _normalize_thresholds(
            self.target_thresholds,
            field_name="target_thresholds",
        )

        try:
            match_mode = RuleMatchMode(
                self.match_mode
            )
        except (TypeError, ValueError) as exc:
            raise HypothesisGeneratorError(
                f"unsupported match mode: {self.match_mode!r}"
            ) from exc

        for field_name in (
            "minimum_total_changes",
            "minimum_causal_depth",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, int):
                raise HypothesisGeneratorError(
                    f"{field_name} must be an integer"
                )

            if value < 0:
                raise HypothesisGeneratorError(
                    f"{field_name} must be non-negative"
                )

        minimum_capacity_loss = (
            _validate_nonnegative_finite(
                self.minimum_capacity_loss,
                field_name="minimum_capacity_loss",
            )
        )
        minimum_capacity_gain = (
            _validate_nonnegative_finite(
                self.minimum_capacity_gain,
                field_name="minimum_capacity_gain",
            )
        )
        minimum_persistence = _validate_unit_interval(
            self.minimum_persistence,
            field_name="minimum_persistence",
        )
        minimum_irreversibility = (
            _validate_unit_interval(
                self.minimum_irreversibility,
                field_name="minimum_irreversibility",
            )
        )

        prior_probability = _validate_unit_interval(
            self.prior_probability,
            field_name="prior_probability",
        )
        validation_score = _validate_unit_interval(
            self.validation_score,
            field_name="validation_score",
        )
        complexity = _validate_nonnegative_finite(
            self.complexity,
            field_name="complexity",
        )
        custom_penalty = _validate_unit_interval(
            self.custom_penalty,
            field_name="custom_penalty",
        )

        if not isinstance(self.enabled, bool):
            raise HypothesisGeneratorError(
                "enabled must be a boolean"
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "hypothesis_name",
            hypothesis_name,
        )
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(
            self,
            "description",
            description,
        )
        object.__setattr__(
            self,
            "initiating_plane_ids",
            initiating_plane_ids,
        )
        object.__setattr__(
            self,
            "initiating_agent_ids",
            initiating_agent_ids,
        )
        object.__setattr__(
            self,
            "candidate_root_ids",
            candidate_root_ids,
        )
        object.__setattr__(
            self,
            "plane_thresholds",
            plane_thresholds,
        )
        object.__setattr__(
            self,
            "agent_thresholds",
            agent_thresholds,
        )
        object.__setattr__(
            self,
            "kind_thresholds",
            kind_thresholds,
        )
        object.__setattr__(
            self,
            "time_scale_thresholds",
            time_scale_thresholds,
        )
        object.__setattr__(
            self,
            "target_thresholds",
            target_thresholds,
        )
        object.__setattr__(
            self,
            "match_mode",
            match_mode,
        )
        object.__setattr__(
            self,
            "minimum_capacity_loss",
            minimum_capacity_loss,
        )
        object.__setattr__(
            self,
            "minimum_capacity_gain",
            minimum_capacity_gain,
        )
        object.__setattr__(
            self,
            "minimum_persistence",
            minimum_persistence,
        )
        object.__setattr__(
            self,
            "minimum_irreversibility",
            minimum_irreversibility,
        )
        object.__setattr__(
            self,
            "prior_probability",
            prior_probability,
        )
        object.__setattr__(
            self,
            "validation_score",
            validation_score,
        )
        object.__setattr__(
            self,
            "complexity",
            complexity,
        )
        object.__setattr__(
            self,
            "custom_penalty",
            custom_penalty,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def has_profile_constraints(self) -> bool:
        return any(
            (
                self.plane_thresholds,
                self.agent_thresholds,
                self.kind_thresholds,
                self.time_scale_thresholds,
                self.target_thresholds,
            )
        )

    def _profile_checks(
        self,
        observed: StructuralSignature,
    ) -> tuple[bool, ...]:
        checks: list[bool] = []

        profile_specs = (
            (
                observed.plane_profile,
                self.plane_thresholds,
            ),
            (
                observed.agent_profile,
                self.agent_thresholds,
            ),
            (
                observed.kind_profile,
                self.kind_thresholds,
            ),
            (
                observed.time_scale_profile,
                self.time_scale_thresholds,
            ),
            (
                observed.target_profile,
                self.target_thresholds,
            ),
        )

        for profile, thresholds in profile_specs:
            for key, threshold in thresholds.items():
                checks.append(
                    float(profile.get(key, 0.0))
                    >= threshold
                )

        return tuple(checks)

    def matches(
        self,
        observed: StructuralSignature,
        *,
        active_plane_ids: Iterable[str] = (),
        known_agent_ids: Iterable[str] = (),
        allowed_root_ids: Iterable[str] = (),
    ) -> bool:
        """Return whether this rule is admissible for the observation."""

        if not isinstance(
            observed,
            StructuralSignature,
        ):
            raise HypothesisGeneratorError(
                "observed must be a StructuralSignature"
            )

        if not self.enabled:
            return False

        if (
            observed.total_changes
            < self.minimum_total_changes
        ):
            return False

        if (
            observed.causal_depth
            < self.minimum_causal_depth
        ):
            return False

        if (
            observed.total_capacity_loss
            < self.minimum_capacity_loss
        ):
            return False

        if (
            observed.total_capacity_gain
            < self.minimum_capacity_gain
        ):
            return False

        if (
            observed.persistence_index
            < self.minimum_persistence
        ):
            return False

        if (
            observed.irreversibility_index
            < self.minimum_irreversibility
        ):
            return False

        active_planes = set(
            _normalize_ids(
                active_plane_ids,
                field_name="active_plane_ids",
            )
        )
        known_agents = set(
            _normalize_ids(
                known_agent_ids,
                field_name="known_agent_ids",
            )
        )
        allowed_roots = set(
            _normalize_ids(
                allowed_root_ids,
                field_name="allowed_root_ids",
            )
        )

        if (
            active_planes
            and self.initiating_plane_ids
            and not (
                active_planes
                & set(self.initiating_plane_ids)
            )
        ):
            return False

        if (
            known_agents
            and self.initiating_agent_ids
            and not (
                known_agents
                & set(self.initiating_agent_ids)
            )
        ):
            return False

        if (
            allowed_roots
            and self.candidate_root_ids
            and not (
                allowed_roots
                & set(self.candidate_root_ids)
            )
        ):
            return False

        checks = self._profile_checks(
            observed
        )

        if not checks:
            return True

        if self.match_mode is RuleMatchMode.ALL:
            return all(checks)

        return any(checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "rule_id": self.rule_id,
            "name": self.name,
            "hypothesis_name": self.hypothesis_name,
            "description": self.description,
            "initiating_plane_ids": list(
                self.initiating_plane_ids
            ),
            "initiating_agent_ids": list(
                self.initiating_agent_ids
            ),
            "candidate_root_ids": list(
                self.candidate_root_ids
            ),
            "plane_thresholds": dict(
                self.plane_thresholds
            ),
            "agent_thresholds": dict(
                self.agent_thresholds
            ),
            "kind_thresholds": dict(
                self.kind_thresholds
            ),
            "time_scale_thresholds": dict(
                self.time_scale_thresholds
            ),
            "target_thresholds": dict(
                self.target_thresholds
            ),
            "match_mode": self.match_mode.value,
            "minimum_total_changes": (
                self.minimum_total_changes
            ),
            "minimum_causal_depth": (
                self.minimum_causal_depth
            ),
            "minimum_capacity_loss": (
                self.minimum_capacity_loss
            ),
            "minimum_capacity_gain": (
                self.minimum_capacity_gain
            ),
            "minimum_persistence": (
                self.minimum_persistence
            ),
            "minimum_irreversibility": (
                self.minimum_irreversibility
            ),
            "prior_probability": (
                self.prior_probability
            ),
            "validation_score": (
                self.validation_score
            ),
            "complexity": self.complexity,
            "custom_penalty": self.custom_penalty,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> CausalRule:
        if not isinstance(data, Mapping):
            raise HypothesisGeneratorError(
                "causal-rule data must be a mapping"
            )

        version = data.get("version", cls.VERSION)

        if version != cls.VERSION:
            raise HypothesisGeneratorError(
                "unsupported causal rule version: "
                f"{version!r}"
            )

        return cls(
            rule_id=str(data["rule_id"]),
            name=str(data["name"]),
            hypothesis_name=str(
                data["hypothesis_name"]
            ),
            description=data.get("description"),
            initiating_plane_ids=tuple(
                data.get("initiating_plane_ids", ())
            ),
            initiating_agent_ids=tuple(
                data.get("initiating_agent_ids", ())
            ),
            candidate_root_ids=tuple(
                data.get("candidate_root_ids", ())
            ),
            plane_thresholds=data.get(
                "plane_thresholds",
                {},
            ),
            agent_thresholds=data.get(
                "agent_thresholds",
                {},
            ),
            kind_thresholds=data.get(
                "kind_thresholds",
                {},
            ),
            time_scale_thresholds=data.get(
                "time_scale_thresholds",
                {},
            ),
            target_thresholds=data.get(
                "target_thresholds",
                {},
            ),
            match_mode=RuleMatchMode(
                data.get(
                    "match_mode",
                    RuleMatchMode.ALL.value,
                )
            ),
            minimum_total_changes=int(
                data.get(
                    "minimum_total_changes",
                    0,
                )
            ),
            minimum_causal_depth=int(
                data.get(
                    "minimum_causal_depth",
                    0,
                )
            ),
            minimum_capacity_loss=float(
                data.get(
                    "minimum_capacity_loss",
                    0.0,
                )
            ),
            minimum_capacity_gain=float(
                data.get(
                    "minimum_capacity_gain",
                    0.0,
                )
            ),
            minimum_persistence=float(
                data.get(
                    "minimum_persistence",
                    0.0,
                )
            ),
            minimum_irreversibility=float(
                data.get(
                    "minimum_irreversibility",
                    0.0,
                )
            ),
            prior_probability=float(
                data.get(
                    "prior_probability",
                    0.5,
                )
            ),
            validation_score=float(
                data.get(
                    "validation_score",
                    0.5,
                )
            ),
            complexity=float(
                data.get("complexity", 1.0)
            ),
            custom_penalty=float(
                data.get(
                    "custom_penalty",
                    0.0,
                )
            ),
            enabled=bool(
                data.get("enabled", True)
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class HypothesisSeed:
    """
    Intermediate candidate before forward prediction.

    A seed records why the candidate was generated and which initiating
    planes, agents, and roots should be tested by the forward model.
    """

    rule_id: str
    name: str

    description: str | None = None
    initiating_plane_ids: tuple[str, ...] = ()
    initiating_agent_ids: tuple[str, ...] = ()
    candidate_root_ids: tuple[str, ...] = ()

    prior_probability: float = 0.5
    validation_score: float = 0.5
    complexity: float = 1.0
    custom_penalty: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    seed_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        rule_id = _require_nonempty_string(
            self.rule_id,
            field_name="rule_id",
        )
        name = _require_nonempty_string(
            self.name,
            field_name="name",
        )
        seed_id = _require_nonempty_string(
            self.seed_id,
            field_name="seed_id",
        )

        description = self.description

        if description is not None:
            description = _require_nonempty_string(
                description,
                field_name="description",
            )

        initiating_plane_ids = _normalize_ids(
            self.initiating_plane_ids,
            field_name="initiating_plane_ids",
        )
        initiating_agent_ids = _normalize_ids(
            self.initiating_agent_ids,
            field_name="initiating_agent_ids",
        )
        candidate_root_ids = _normalize_ids(
            self.candidate_root_ids,
            field_name="candidate_root_ids",
        )

        prior_probability = _validate_unit_interval(
            self.prior_probability,
            field_name="prior_probability",
        )
        validation_score = _validate_unit_interval(
            self.validation_score,
            field_name="validation_score",
        )
        complexity = _validate_nonnegative_finite(
            self.complexity,
            field_name="complexity",
        )
        custom_penalty = _validate_unit_interval(
            self.custom_penalty,
            field_name="custom_penalty",
        )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "seed_id", seed_id)
        object.__setattr__(
            self,
            "description",
            description,
        )
        object.__setattr__(
            self,
            "initiating_plane_ids",
            initiating_plane_ids,
        )
        object.__setattr__(
            self,
            "initiating_agent_ids",
            initiating_agent_ids,
        )
        object.__setattr__(
            self,
            "candidate_root_ids",
            candidate_root_ids,
        )
        object.__setattr__(
            self,
            "prior_probability",
            prior_probability,
        )
        object.__setattr__(
            self,
            "validation_score",
            validation_score,
        )
        object.__setattr__(
            self,
            "complexity",
            complexity,
        )
        object.__setattr__(
            self,
            "custom_penalty",
            custom_penalty,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "seed_id": self.seed_id,
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "initiating_plane_ids": list(
                self.initiating_plane_ids
            ),
            "initiating_agent_ids": list(
                self.initiating_agent_ids
            ),
            "candidate_root_ids": list(
                self.candidate_root_ids
            ),
            "prior_probability": (
                self.prior_probability
            ),
            "validation_score": (
                self.validation_score
            ),
            "complexity": self.complexity,
            "custom_penalty": self.custom_penalty,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    """Result of processing one hypothesis seed."""

    seed: HypothesisSeed
    status: CandidateStatus
    hypothesis: HistoryHypothesis | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.seed,
            HypothesisSeed,
        ):
            raise HypothesisGeneratorError(
                "seed must be a HypothesisSeed"
            )

        try:
            status = CandidateStatus(
                self.status
            )
        except (TypeError, ValueError) as exc:
            raise HypothesisGeneratorError(
                f"unsupported candidate status: {self.status!r}"
            ) from exc

        if (
            self.hypothesis is not None
            and not isinstance(
                self.hypothesis,
                HistoryHypothesis,
            )
        ):
            raise HypothesisGeneratorError(
                "hypothesis must be a HistoryHypothesis"
            )

        error_message = self.error_message

        if error_message is not None:
            error_message = _require_nonempty_string(
                error_message,
                field_name="error_message",
            )

        if (
            status is CandidateStatus.PREDICTED
            and self.hypothesis is None
        ):
            raise HypothesisGeneratorError(
                "predicted candidates require a hypothesis"
            )

        if (
            status is CandidateStatus.FAILED
            and error_message is None
        ):
            raise HypothesisGeneratorError(
                "failed candidates require an error message"
            )

        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "error_message",
            error_message,
        )


@dataclass(frozen=True, slots=True)
class HypothesisGenerationResult:
    """Immutable output of one candidate-generation run."""

    observed_signature_id: str
    candidates: tuple[GeneratedCandidate, ...] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    result_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        observed_signature_id = _require_nonempty_string(
            self.observed_signature_id,
            field_name="observed_signature_id",
        )
        result_id = _require_nonempty_string(
            self.result_id,
            field_name="result_id",
        )

        candidates = tuple(self.candidates)

        if not all(
            isinstance(item, GeneratedCandidate)
            for item in candidates
        ):
            raise HypothesisGeneratorError(
                "all candidates must be GeneratedCandidate objects"
            )

        seed_ids = [
            item.seed.seed_id
            for item in candidates
        ]

        if len(seed_ids) != len(set(seed_ids)):
            raise HypothesisGeneratorError(
                "seed_id values must be unique"
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        object.__setattr__(
            self,
            "observed_signature_id",
            observed_signature_id,
        )
        object.__setattr__(
            self,
            "result_id",
            result_id,
        )
        object.__setattr__(
            self,
            "candidates",
            candidates,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )

    @property
    def hypotheses(
        self,
    ) -> tuple[HistoryHypothesis, ...]:
        return tuple(
            item.hypothesis
            for item in self.candidates
            if item.hypothesis is not None
        )

    @property
    def failed(
        self,
    ) -> tuple[GeneratedCandidate, ...]:
        return tuple(
            item
            for item in self.candidates
            if item.status is CandidateStatus.FAILED
        )

    @property
    def predicted_count(self) -> int:
        return len(self.hypotheses)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "result_id": self.result_id,
            "observed_signature_id": (
                self.observed_signature_id
            ),
            "predicted_count": self.predicted_count,
            "failed_count": self.failed_count,
            "candidates": [
                {
                    "seed": item.seed.to_dict(),
                    "status": item.status.value,
                    "hypothesis": (
                        None
                        if item.hypothesis is None
                        else item.hypothesis.to_dict()
                    ),
                    "error_message": (
                        item.error_message
                    ),
                }
                for item in self.candidates
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HypothesisGenerator:
    """
    Generate candidate histories from explicit causal rules.

    max_candidates:
        Maximum number of matched rules processed in one call.

    fail_fast:
        When True, predictor exceptions stop generation.
        When False, failures are recorded and remaining rules continue.
    """

    rules: tuple[CausalRule, ...]
    max_candidates: int = 100
    fail_fast: bool = False

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        rules = tuple(self.rules)

        if not all(
            isinstance(rule, CausalRule)
            for rule in rules
        ):
            raise HypothesisGeneratorError(
                "all rules must be CausalRule objects"
            )

        rule_ids = [
            rule.rule_id
            for rule in rules
        ]

        if len(rule_ids) != len(set(rule_ids)):
            raise HypothesisGeneratorError(
                "rule_id values must be unique"
            )

        if not isinstance(
            self.max_candidates,
            int,
        ):
            raise HypothesisGeneratorError(
                "max_candidates must be an integer"
            )

        if self.max_candidates <= 0:
            raise HypothesisGeneratorError(
                "max_candidates must be positive"
            )

        if not isinstance(self.fail_fast, bool):
            raise HypothesisGeneratorError(
                "fail_fast must be a boolean"
            )

        ordered_rules = tuple(
            sorted(
                rules,
                key=lambda rule: (
                    -rule.prior_probability,
                    rule.complexity,
                    rule.rule_id,
                ),
            )
        )

        object.__setattr__(
            self,
            "rules",
            ordered_rules,
        )

    def matching_rules(
        self,
        observed: StructuralSignature,
        *,
        active_plane_ids: Iterable[str] = (),
        known_agent_ids: Iterable[str] = (),
        allowed_root_ids: Iterable[str] = (),
    ) -> tuple[CausalRule, ...]:
        """Return enabled rules admissible for the observation."""

        if not isinstance(
            observed,
            StructuralSignature,
        ):
            raise HypothesisGeneratorError(
                "observed must be a StructuralSignature"
            )

        matched = tuple(
            rule
            for rule in self.rules
            if rule.matches(
                observed,
                active_plane_ids=active_plane_ids,
                known_agent_ids=known_agent_ids,
                allowed_root_ids=allowed_root_ids,
            )
        )

        return matched[
            : self.max_candidates
        ]

    @staticmethod
    def seed_from_rule(
        rule: CausalRule,
        observed: StructuralSignature,
    ) -> HypothesisSeed:
        """Convert one matched causal rule into a prediction seed."""

        if not isinstance(rule, CausalRule):
            raise HypothesisGeneratorError(
                "rule must be a CausalRule"
            )

        if not isinstance(
            observed,
            StructuralSignature,
        ):
            raise HypothesisGeneratorError(
                "observed must be a StructuralSignature"
            )

        metadata = {
            "source_rule_name": rule.name,
            "observed_signature_id": (
                observed.signature_id
            ),
            "observed_pattern_id": (
                observed.source_pattern_id
            ),
        }
        metadata.update(
            dict(rule.metadata)
        )

        return HypothesisSeed(
            rule_id=rule.rule_id,
            name=rule.hypothesis_name,
            description=rule.description,
            initiating_plane_ids=(
                rule.initiating_plane_ids
            ),
            initiating_agent_ids=(
                rule.initiating_agent_ids
            ),
            candidate_root_ids=(
                rule.candidate_root_ids
            ),
            prior_probability=(
                rule.prior_probability
            ),
            validation_score=(
                rule.validation_score
            ),
            complexity=rule.complexity,
            custom_penalty=rule.custom_penalty,
            metadata=metadata,
        )

    def generate_seeds(
        self,
        observed: StructuralSignature,
        *,
        active_plane_ids: Iterable[str] = (),
        known_agent_ids: Iterable[str] = (),
        allowed_root_ids: Iterable[str] = (),
    ) -> tuple[HypothesisSeed, ...]:
        """Generate deterministic candidate seeds."""

        rules = self.matching_rules(
            observed,
            active_plane_ids=active_plane_ids,
            known_agent_ids=known_agent_ids,
            allowed_root_ids=allowed_root_ids,
        )

        return tuple(
            self.seed_from_rule(
                rule,
                observed,
            )
            for rule in rules
        )

    def generate(
        self,
        observed: StructuralSignature,
        predictor: SignaturePredictor
        | Callable[
            [HypothesisSeed, StructuralSignature],
            StructuralSignature,
        ],
        *,
        active_plane_ids: Iterable[str] = (),
        known_agent_ids: Iterable[str] = (),
        allowed_root_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> HypothesisGenerationResult:
        """
        Generate seeds, call the forward predictor, and build hypotheses.
        """

        if not isinstance(
            observed,
            StructuralSignature,
        ):
            raise HypothesisGeneratorError(
                "observed must be a StructuralSignature"
            )

        if not callable(predictor):
            raise HypothesisGeneratorError(
                "predictor must be callable"
            )

        seeds = self.generate_seeds(
            observed,
            active_plane_ids=active_plane_ids,
            known_agent_ids=known_agent_ids,
            allowed_root_ids=allowed_root_ids,
        )

        candidates: list[GeneratedCandidate] = []

        for seed in seeds:
            try:
                predicted_signature = predictor(
                    seed,
                    observed,
                )

                if not isinstance(
                    predicted_signature,
                    StructuralSignature,
                ):
                    raise HypothesisGeneratorError(
                        "predictor must return a "
                        "StructuralSignature"
                    )

                hypothesis = HistoryHypothesis(
                    hypothesis_id=seed.seed_id,
                    name=seed.name,
                    description=seed.description,
                    predicted_signature=(
                        predicted_signature
                    ),
                    initiating_plane_ids=(
                        seed.initiating_plane_ids
                    ),
                    initiating_agent_ids=(
                        seed.initiating_agent_ids
                    ),
                    candidate_root_ids=(
                        seed.candidate_root_ids
                    ),
                    prior_probability=(
                        seed.prior_probability
                    ),
                    validation_score=(
                        seed.validation_score
                    ),
                    complexity=seed.complexity,
                    custom_penalty=(
                        seed.custom_penalty
                    ),
                    status=(
                        HypothesisStatus.FORWARD_VALIDATED
                    ),
                    metadata={
                        **dict(seed.metadata),
                        "generator_version": (
                            self.VERSION
                        ),
                        "seed_id": seed.seed_id,
                        "rule_id": seed.rule_id,
                    },
                )

                candidates.append(
                    GeneratedCandidate(
                        seed=seed,
                        status=CandidateStatus.PREDICTED,
                        hypothesis=hypothesis,
                    )
                )

            except Exception as exc:
                if self.fail_fast:
                    raise

                candidates.append(
                    GeneratedCandidate(
                        seed=seed,
                        status=CandidateStatus.FAILED,
                        error_message=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )

        merged_metadata = {
            "generator_version": self.VERSION,
            "rule_count": len(self.rules),
            "matched_rule_count": len(seeds),
            "max_candidates": self.max_candidates,
        }

        if metadata is not None:
            merged_metadata.update(
                dict(metadata)
            )

        return HypothesisGenerationResult(
            observed_signature_id=(
                observed.signature_id
            ),
            candidates=tuple(candidates),
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "rules": [
                rule.to_dict()
                for rule in self.rules
            ],
            "max_candidates": self.max_candidates,
            "fail_fast": self.fail_fast,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> HypothesisGenerator:
        if not isinstance(data, Mapping):
            raise HypothesisGeneratorError(
                "hypothesis-generator data must be a mapping"
            )

        version = data.get("version", cls.VERSION)

        if version != cls.VERSION:
            raise HypothesisGeneratorError(
                "unsupported hypothesis generator version: "
                f"{version!r}"
            )

        raw_rules = data.get("rules", ())

        if not isinstance(
            raw_rules,
            (list, tuple),
        ):
            raise HypothesisGeneratorError(
                "rules must be a sequence"
            )

        return cls(
            rules=tuple(
                CausalRule.from_dict(item)
                for item in raw_rules
            ),
            max_candidates=int(
                data.get(
                    "max_candidates",
                    100,
                )
            ),
            fail_fast=bool(
                data.get(
                    "fail_fast",
                    False,
                )
            ),
        )


__all__ = [
    "CandidateStatus",
    "CausalRule",
    "GeneratedCandidate",
    "HypothesisGenerationResult",
    "HypothesisGenerator",
    "HypothesisGeneratorError",
    "HypothesisSeed",
    "RuleMatchMode",
    "SignaturePredictor",
]