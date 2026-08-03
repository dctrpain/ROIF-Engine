from __future__ import annotations

"""
Aggregated structural memory for the ROIF Engine.

HistoryEvent answers:

    What happened?

IrreversibleChange answers:

    What remained after one event?

HistoryPattern answers:

    What does the complete set of retained changes mean for the structure?

A HistoryPattern is an immutable, causally validated collection of
IrreversibleChange objects. It provides:

- chronological ordering;
- causal ancestry and descendants;
- capacity-loss and capacity-gain summaries;
- dominant planes, agents, targets, kinds, and time scales;
- persistence and irreversibility indices;
- progression and adaptation indicators;
- filtering and merging;
- serialization;
- a stable basis for future StructuralSignature and HistoryDecoder modules.

HistoryPattern does not modify the physical structure. It describes the
persistent memory already recorded in that structure.
"""

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .event import (
    HistoryTargetKind,
    TimeScale,
)
from .irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeKind,
    TracePersistence,
)


class HistoryPatternError(ValueError):
    """Raised when a HistoryPattern is invalid or inconsistent."""


def _require_nonempty_string(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise HistoryPatternError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise HistoryPatternError(
            f"{field_name} must not be empty"
        )

    return normalized


def _validate_nonnegative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoryPatternError(
            f"{field_name} must be a real number"
        ) from exc

    if not isfinite(numeric):
        raise HistoryPatternError(
            f"{field_name} must be finite"
        )

    if numeric < 0.0:
        raise HistoryPatternError(
            f"{field_name} must be non-negative"
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
        raise HistoryPatternError(
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


@dataclass(frozen=True, slots=True)
class RankedContribution:
    """
    One ranked contribution to a HistoryPattern.

    score is normalized to [0, 1] relative to the total weighted contribution
    for the requested category.
    """

    key: str
    count: int
    raw_weight: float
    score: float

    def __post_init__(self) -> None:
        key = _require_nonempty_string(
            self.key,
            field_name="key",
        )

        if not isinstance(self.count, int):
            raise HistoryPatternError(
                "count must be an integer"
            )

        if self.count < 0:
            raise HistoryPatternError(
                "count must be non-negative"
            )

        raw_weight = _validate_nonnegative_finite(
            self.raw_weight,
            field_name="raw_weight",
        )
        score = _validate_nonnegative_finite(
            self.score,
            field_name="score",
        )

        if score > 1.0:
            raise HistoryPatternError(
                "score must be in [0, 1]"
            )

        object.__setattr__(self, "key", key)
        object.__setattr__(
            self,
            "raw_weight",
            raw_weight,
        )
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "count": self.count,
            "raw_weight": self.raw_weight,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class HistoryPattern:
    """
    Immutable causal pattern of persistent structural changes.

    Parameters
    ----------
    changes:
        Persistent structural changes belonging to this pattern.

    label:
        Optional human-readable name.

    description:
        Optional explanation of the pattern.

    metadata:
        Additional domain-specific information.

    pattern_id:
        Stable identifier of the pattern.

    Strict causal validation
    ------------------------
    Every cause_change_id must either:

    - refer to another change inside the same pattern; or
    - be listed in external_cause_ids.

    This makes partially observed histories possible without silently
    accepting broken causal references.
    """

    changes: tuple[IrreversibleChange, ...] = ()
    label: str | None = None
    description: str | None = None
    external_cause_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    pattern_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    VERSION = "1.0.0"

    def __post_init__(self) -> None:
        changes = tuple(self.changes)

        if not all(
            isinstance(change, IrreversibleChange)
            for change in changes
        ):
            raise HistoryPatternError(
                "all changes must be IrreversibleChange objects"
            )

        change_ids = [
            change.change_id
            for change in changes
        ]

        if len(change_ids) != len(set(change_ids)):
            raise HistoryPatternError(
                "change_id values must be unique"
            )

        ordered_changes = tuple(
            sorted(
                changes,
                key=lambda item: (
                    item.onset_time,
                    item.recorded_time,
                    item.change_id,
                ),
            )
        )

        external_cause_ids = self._normalize_ids(
            self.external_cause_ids,
            field_name="external_cause_ids",
        )

        internal_ids = {
            change.change_id
            for change in ordered_changes
        }
        external_ids = set(external_cause_ids)

        overlap = internal_ids & external_ids

        if overlap:
            raise HistoryPatternError(
                "an internal change cannot also be an "
                "external cause"
            )

        for change in ordered_changes:
            for cause_id in change.cause_change_ids:
                if (
                    cause_id not in internal_ids
                    and cause_id not in external_ids
                ):
                    raise HistoryPatternError(
                        "unknown causal reference "
                        f"{cause_id!r} in change "
                        f"{change.change_id!r}"
                    )

        self._validate_causal_time_order(
            ordered_changes
        )
        self._validate_acyclic(
            ordered_changes
        )

        label = self.label

        if label is not None:
            label = _require_nonempty_string(
                label,
                field_name="label",
            )

        description = self.description

        if description is not None:
            description = _require_nonempty_string(
                description,
                field_name="description",
            )

        metadata = _freeze_mapping(
            self.metadata,
            field_name="metadata",
        )

        pattern_id = _require_nonempty_string(
            self.pattern_id,
            field_name="pattern_id",
        )

        object.__setattr__(
            self,
            "changes",
            ordered_changes,
        )
        object.__setattr__(
            self,
            "external_cause_ids",
            external_cause_ids,
        )
        object.__setattr__(
            self,
            "label",
            label,
        )
        object.__setattr__(
            self,
            "description",
            description,
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )
        object.__setattr__(
            self,
            "pattern_id",
            pattern_id,
        )

    @staticmethod
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

    @staticmethod
    def _validate_causal_time_order(
        changes: tuple[IrreversibleChange, ...],
    ) -> None:
        by_id = {
            change.change_id: change
            for change in changes
        }

        for change in changes:
            for cause_id in change.cause_change_ids:
                cause = by_id.get(cause_id)

                if cause is None:
                    continue

                if cause.onset_time > change.onset_time:
                    raise HistoryPatternError(
                        "a cause cannot begin after its effect: "
                        f"{cause_id!r} -> {change.change_id!r}"
                    )

                if (
                    cause.onset_time == change.onset_time
                    and cause.recorded_time
                    > change.recorded_time
                ):
                    raise HistoryPatternError(
                        "a cause cannot be recorded after an "
                        "effect at the same onset time"
                    )

    @staticmethod
    def _validate_acyclic(
        changes: tuple[IrreversibleChange, ...],
    ) -> None:
        internal_ids = {
            change.change_id
            for change in changes
        }

        indegree: dict[str, int] = {
            change_id: 0
            for change_id in internal_ids
        }
        children: dict[str, list[str]] = {
            change_id: []
            for change_id in internal_ids
        }

        for change in changes:
            for cause_id in change.cause_change_ids:
                if cause_id not in internal_ids:
                    continue

                children[cause_id].append(
                    change.change_id
                )
                indegree[change.change_id] += 1

        queue = deque(
            sorted(
                change_id
                for change_id, degree
                in indegree.items()
                if degree == 0
            )
        )

        visited = 0

        while queue:
            current = queue.popleft()
            visited += 1

            for child in sorted(
                children[current]
            ):
                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        if visited != len(internal_ids):
            raise HistoryPatternError(
                "causal graph contains a cycle"
            )

    def __len__(self) -> int:
        return len(self.changes)

    def __iter__(
        self,
    ) -> Iterator[IrreversibleChange]:
        return iter(self.changes)

    def __contains__(
        self,
        change_id: object,
    ) -> bool:
        if not isinstance(change_id, str):
            return False

        return self.change_by_id(
            change_id
        ) is not None

    @property
    def is_empty(self) -> bool:
        return not self.changes

    @property
    def total_changes(self) -> int:
        return len(self.changes)

    @property
    def first_change(
        self,
    ) -> IrreversibleChange | None:
        if not self.changes:
            return None

        return self.changes[0]

    @property
    def last_change(
        self,
    ) -> IrreversibleChange | None:
        if not self.changes:
            return None

        return max(
            self.changes,
            key=lambda item: (
                item.recorded_time,
                item.onset_time,
                item.change_id,
            ),
        )

    @property
    def start_time(self) -> float:
        first = self.first_change

        if first is None:
            return 0.0

        return first.onset_time

    @property
    def end_time(self) -> float:
        last = self.last_change

        if last is None:
            return 0.0

        return last.recorded_time

    @property
    def duration(self) -> float:
        if self.is_empty:
            return 0.0

        return self.end_time - self.start_time

    @property
    def total_capacity_loss(self) -> float:
        return sum(
            -change.capacity_effect
            for change in self.changes
            if change.capacity_effect < 0.0
        )

    @property
    def total_capacity_gain(self) -> float:
        return sum(
            change.capacity_effect
            for change in self.changes
            if change.capacity_effect > 0.0
        )

    @property
    def net_capacity_effect(self) -> float:
        return sum(
            change.capacity_effect
            for change in self.changes
        )

    @property
    def cumulative_signature_weight(self) -> float:
        return sum(
            change.signature_weight
            for change in self.changes
        )

    @property
    def mean_signature_weight(self) -> float:
        if not self.changes:
            return 0.0

        return (
            self.cumulative_signature_weight
            / len(self.changes)
        )

    @property
    def persistence_index(self) -> float:
        """
        Weighted mean persistence in [0, 1].

        Permanent changes receive full weight, long-lived changes receive
        partial weight, and transient traces receive lower weight.
        """

        if not self.changes:
            return 0.0

        permanence_factor = {
            TracePersistence.TRANSIENT: 0.25,
            TracePersistence.LONG_LIVED: 0.70,
            TracePersistence.PERMANENT: 1.00,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for change in self.changes:
            trace_weight = max(
                change.signature_weight,
                1e-12,
            )
            weighted_sum += (
                permanence_factor[change.permanence]
                * trace_weight
            )
            total_weight += trace_weight

        return _safe_ratio(
            weighted_sum,
            total_weight,
        )

    @property
    def irreversibility_index(self) -> float:
        """
        Weighted irreversibility in [0, 1].

        This is not a damage score. Beneficial adaptation can also be highly
        irreversible.
        """

        if not self.changes:
            return 0.0

        reversibility_factor = {
            "reversible": 0.0,
            "partially_reversible": 0.5,
            "irreversible": 1.0,
            "unknown": 0.5,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for change in self.changes:
            weight = max(
                change.signature_weight,
                1e-12,
            )
            weighted_sum += (
                reversibility_factor[
                    change.reversibility.value
                ]
                * weight
            )
            total_weight += weight

        return _safe_ratio(
            weighted_sum,
            total_weight,
        )

    @property
    def harmful_weight(self) -> float:
        return sum(
            change.signature_weight
            for change in self.changes
            if (
                change.functional_effect
                is FunctionalEffect.HARMFUL
            )
        )

    @property
    def beneficial_weight(self) -> float:
        return sum(
            change.signature_weight
            for change in self.changes
            if (
                change.functional_effect
                is FunctionalEffect.BENEFICIAL
            )
        )

    @property
    def mixed_weight(self) -> float:
        return sum(
            change.signature_weight
            for change in self.changes
            if (
                change.functional_effect
                is FunctionalEffect.MIXED
            )
        )

    @property
    def progression_index(self) -> float:
        """
        Relative harmful dominance in [0, 1].

        0 means no harmful dominance.
        1 means the weighted pattern is entirely harmful.
        """

        relevant = (
            self.harmful_weight
            + self.beneficial_weight
            + self.mixed_weight
        )

        harmful_equivalent = (
            self.harmful_weight
            + 0.5 * self.mixed_weight
        )

        return _safe_ratio(
            harmful_equivalent,
            relevant,
        )

    @property
    def adaptation_index(self) -> float:
        """
        Relative beneficial dominance in [0, 1].
        """

        relevant = (
            self.harmful_weight
            + self.beneficial_weight
            + self.mixed_weight
        )

        beneficial_equivalent = (
            self.beneficial_weight
            + 0.5 * self.mixed_weight
        )

        return _safe_ratio(
            beneficial_equivalent,
            relevant,
        )

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
    def root_changes(
        self,
    ) -> tuple[IrreversibleChange, ...]:
        internal_ids = {
            change.change_id
            for change in self.changes
        }

        roots = []

        for change in self.changes:
            internal_causes = [
                cause_id
                for cause_id in change.cause_change_ids
                if cause_id in internal_ids
            ]

            if not internal_causes:
                roots.append(change)

        return tuple(roots)

    @property
    def leaf_changes(
        self,
    ) -> tuple[IrreversibleChange, ...]:
        referenced_ids = {
            cause_id
            for change in self.changes
            for cause_id in change.cause_change_ids
        }

        return tuple(
            change
            for change in self.changes
            if change.change_id
            not in referenced_ids
        )

    @property
    def causal_depth(self) -> int:
        """
        Return the maximum number of internal changes in one causal chain.
        """

        if not self.changes:
            return 0

        internal_ids = {
            change.change_id
            for change in self.changes
        }

        parents = {
            change.change_id: tuple(
                cause_id
                for cause_id in change.cause_change_ids
                if cause_id in internal_ids
            )
            for change in self.changes
        }

        memo: dict[str, int] = {}

        def depth(change_id: str) -> int:
            if change_id in memo:
                return memo[change_id]

            causes = parents[change_id]

            if not causes:
                result = 1
            else:
                result = 1 + max(
                    depth(cause_id)
                    for cause_id in causes
                )

            memo[change_id] = result
            return result

        return max(
            depth(change.change_id)
            for change in self.changes
        )

    def change_by_id(
        self,
        change_id: str,
    ) -> IrreversibleChange | None:
        normalized = _require_nonempty_string(
            change_id,
            field_name="change_id",
        )

        for change in self.changes:
            if change.change_id == normalized:
                return change

        return None

    def chronology(
        self,
    ) -> tuple[IrreversibleChange, ...]:
        return self.changes

    def direct_causes_of(
        self,
        change_id: str,
    ) -> tuple[IrreversibleChange, ...]:
        change = self.change_by_id(change_id)

        if change is None:
            raise HistoryPatternError(
                f"unknown change_id: {change_id!r}"
            )

        by_id = {
            item.change_id: item
            for item in self.changes
        }

        return tuple(
            by_id[cause_id]
            for cause_id in change.cause_change_ids
            if cause_id in by_id
        )

    def direct_effects_of(
        self,
        change_id: str,
    ) -> tuple[IrreversibleChange, ...]:
        normalized = _require_nonempty_string(
            change_id,
            field_name="change_id",
        )

        if self.change_by_id(normalized) is None:
            raise HistoryPatternError(
                f"unknown change_id: {normalized!r}"
            )

        return tuple(
            change
            for change in self.changes
            if normalized in change.cause_change_ids
        )

    def ancestors_of(
        self,
        change_id: str,
    ) -> tuple[IrreversibleChange, ...]:
        target = self.change_by_id(change_id)

        if target is None:
            raise HistoryPatternError(
                f"unknown change_id: {change_id!r}"
            )

        by_id = {
            change.change_id: change
            for change in self.changes
        }

        visited: set[str] = set()
        stack = list(target.cause_change_ids)

        while stack:
            current_id = stack.pop()

            if current_id in visited:
                continue

            current = by_id.get(current_id)

            if current is None:
                continue

            visited.add(current_id)
            stack.extend(
                current.cause_change_ids
            )

        return tuple(
            change
            for change in self.changes
            if change.change_id in visited
        )

    def descendants_of(
        self,
        change_id: str,
    ) -> tuple[IrreversibleChange, ...]:
        normalized = _require_nonempty_string(
            change_id,
            field_name="change_id",
        )

        if self.change_by_id(normalized) is None:
            raise HistoryPatternError(
                f"unknown change_id: {normalized!r}"
            )

        children: dict[str, list[str]] = defaultdict(
            list
        )

        for change in self.changes:
            for cause_id in change.cause_change_ids:
                children[cause_id].append(
                    change.change_id
                )

        visited: set[str] = set()
        stack = list(children[normalized])

        while stack:
            current_id = stack.pop()

            if current_id in visited:
                continue

            visited.add(current_id)
            stack.extend(
                children[current_id]
            )

        return tuple(
            change
            for change in self.changes
            if change.change_id in visited
        )

    def _ranked_contributions(
        self,
        key_provider,
    ) -> tuple[RankedContribution, ...]:
        counts: Counter[str] = Counter()
        weights: defaultdict[str, float] = defaultdict(
            float
        )

        for change in self.changes:
            keys = tuple(key_provider(change))

            for key in keys:
                normalized = _require_nonempty_string(
                    str(key),
                    field_name="contribution key",
                )
                counts[normalized] += 1
                weights[normalized] += (
                    change.signature_weight
                )

        total_weight = sum(weights.values())

        ranked = [
            RankedContribution(
                key=key,
                count=counts[key],
                raw_weight=weights[key],
                score=_safe_ratio(
                    weights[key],
                    total_weight,
                ),
            )
            for key in weights
        ]

        ranked.sort(
            key=lambda item: (
                -item.raw_weight,
                -item.count,
                item.key,
            )
        )

        return tuple(ranked)

    def plane_profile(
        self,
    ) -> tuple[RankedContribution, ...]:
        return self._ranked_contributions(
            lambda change: change.plane_ids
        )

    def agent_profile(
        self,
    ) -> tuple[RankedContribution, ...]:
        return self._ranked_contributions(
            lambda change: change.agent_ids
        )

    def kind_profile(
        self,
    ) -> tuple[RankedContribution, ...]:
        return self._ranked_contributions(
            lambda change: (
                change.kind.value,
            )
        )

    def time_scale_profile(
        self,
    ) -> tuple[RankedContribution, ...]:
        return self._ranked_contributions(
            lambda change: (
                change.time_scale.value,
            )
        )

    def target_profile(
        self,
    ) -> tuple[RankedContribution, ...]:
        return self._ranked_contributions(
            lambda change: (
                (
                    f"{change.target.kind.value}:"
                    f"{change.target.target_id}"
                ),
            )
        )

    @property
    def dominant_plane(
        self,
    ) -> RankedContribution | None:
        profile = self.plane_profile()
        return profile[0] if profile else None

    @property
    def dominant_agent(
        self,
    ) -> RankedContribution | None:
        profile = self.agent_profile()
        return profile[0] if profile else None

    @property
    def dominant_kind(
        self,
    ) -> RankedContribution | None:
        profile = self.kind_profile()
        return profile[0] if profile else None

    @property
    def dominant_time_scale(
        self,
    ) -> RankedContribution | None:
        profile = self.time_scale_profile()
        return profile[0] if profile else None

    def filter_by_plane(
        self,
        plane_id: str,
    ) -> HistoryPattern:
        normalized = _require_nonempty_string(
            plane_id,
            field_name="plane_id",
        )

        return self._filtered_pattern(
            change
            for change in self.changes
            if normalized in change.plane_ids
        )

    def filter_by_agent(
        self,
        agent_id: str,
    ) -> HistoryPattern:
        normalized = _require_nonempty_string(
            agent_id,
            field_name="agent_id",
        )

        return self._filtered_pattern(
            change
            for change in self.changes
            if normalized in change.agent_ids
        )

    def filter_by_kind(
        self,
        kind: IrreversibleChangeKind | str,
    ) -> HistoryPattern:
        try:
            normalized = IrreversibleChangeKind(
                kind
            )
        except (TypeError, ValueError) as exc:
            raise HistoryPatternError(
                f"unsupported change kind: {kind!r}"
            ) from exc

        return self._filtered_pattern(
            change
            for change in self.changes
            if change.kind is normalized
        )

    def filter_by_time_scale(
        self,
        time_scale: TimeScale | str,
    ) -> HistoryPattern:
        try:
            normalized = TimeScale(
                time_scale
            )
        except (TypeError, ValueError) as exc:
            raise HistoryPatternError(
                f"unsupported time scale: {time_scale!r}"
            ) from exc

        return self._filtered_pattern(
            change
            for change in self.changes
            if change.time_scale is normalized
        )

    def filter_by_target_kind(
        self,
        target_kind: HistoryTargetKind | str,
    ) -> HistoryPattern:
        try:
            normalized = HistoryTargetKind(
                target_kind
            )
        except (TypeError, ValueError) as exc:
            raise HistoryPatternError(
                "unsupported target kind: "
                f"{target_kind!r}"
            ) from exc

        return self._filtered_pattern(
            change
            for change in self.changes
            if change.target.kind is normalized
        )

    def filter_by_time(
        self,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> HistoryPattern:
        if start is None:
            start_value = 0.0
        else:
            start_value = _validate_nonnegative_finite(
                start,
                field_name="start",
            )

        if end is None:
            end_value = float("inf")
        else:
            end_value = _validate_nonnegative_finite(
                end,
                field_name="end",
            )

        if end_value < start_value:
            raise HistoryPatternError(
                "end must not precede start"
            )

        return self._filtered_pattern(
            change
            for change in self.changes
            if (
                change.onset_time >= start_value
                and change.recorded_time <= end_value
            )
        )

    def _filtered_pattern(
        self,
        selected: Iterable[IrreversibleChange],
    ) -> HistoryPattern:
        selected_changes = tuple(selected)
        selected_ids = {
            change.change_id
            for change in selected_changes
        }

        external_ids = set(
            self.external_cause_ids
        )

        for change in selected_changes:
            for cause_id in change.cause_change_ids:
                if cause_id not in selected_ids:
                    external_ids.add(cause_id)

        return HistoryPattern(
            changes=selected_changes,
            label=self.label,
            description=self.description,
            external_cause_ids=tuple(
                sorted(external_ids)
            ),
            metadata=self.metadata,
        )

    def append(
        self,
        change: IrreversibleChange,
    ) -> HistoryPattern:
        if not isinstance(
            change,
            IrreversibleChange,
        ):
            raise HistoryPatternError(
                "change must be an IrreversibleChange"
            )

        return HistoryPattern(
            changes=(
                *self.changes,
                change,
            ),
            label=self.label,
            description=self.description,
            external_cause_ids=(
                self.external_cause_ids
            ),
            metadata=self.metadata,
            pattern_id=self.pattern_id,
        )

    def merge(
        self,
        other: HistoryPattern,
        *,
        label: str | None = None,
        description: str | None = None,
    ) -> HistoryPattern:
        if not isinstance(
            other,
            HistoryPattern,
        ):
            raise HistoryPatternError(
                "other must be a HistoryPattern"
            )

        by_id: dict[str, IrreversibleChange] = {
            change.change_id: change
            for change in self.changes
        }

        for change in other.changes:
            existing = by_id.get(
                change.change_id
            )

            if (
                existing is not None
                and existing != change
            ):
                raise HistoryPatternError(
                    "conflicting changes share change_id "
                    f"{change.change_id!r}"
                )

            by_id[change.change_id] = change

        external_ids = (
            set(self.external_cause_ids)
            | set(other.external_cause_ids)
        )
        internal_ids = set(by_id)
        external_ids -= internal_ids

        merged_metadata = dict(self.metadata)

        for key, value in other.metadata.items():
            if (
                key in merged_metadata
                and merged_metadata[key] != value
            ):
                raise HistoryPatternError(
                    "conflicting metadata value for "
                    f"{key!r}"
                )

            merged_metadata[key] = value

        return HistoryPattern(
            changes=tuple(by_id.values()),
            label=(
                label
                if label is not None
                else self.label or other.label
            ),
            description=(
                description
                if description is not None
                else (
                    self.description
                    or other.description
                )
            ),
            external_cause_ids=tuple(
                sorted(external_ids)
            ),
            metadata=merged_metadata,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "label": self.label,
            "total_changes": self.total_changes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "root_change_ids": [
                change.change_id
                for change in self.root_changes
            ],
            "leaf_change_ids": [
                change.change_id
                for change in self.leaf_changes
            ],
            "causal_depth": self.causal_depth,
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
            "is_progressive": self.is_progressive,
            "is_adaptive": self.is_adaptive,
            "is_balanced": self.is_balanced,
            "dominant_plane": (
                None
                if self.dominant_plane is None
                else self.dominant_plane.to_dict()
            ),
            "dominant_agent": (
                None
                if self.dominant_agent is None
                else self.dominant_agent.to_dict()
            ),
            "dominant_kind": (
                None
                if self.dominant_kind is None
                else self.dominant_kind.to_dict()
            ),
            "dominant_time_scale": (
                None
                if self.dominant_time_scale is None
                else self.dominant_time_scale.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "pattern_id": self.pattern_id,
            "label": self.label,
            "description": self.description,
            "external_cause_ids": list(
                self.external_cause_ids
            ),
            "changes": [
                change.to_dict()
                for change in self.changes
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> HistoryPattern:
        if not isinstance(data, Mapping):
            raise HistoryPatternError(
                "history pattern data must be a mapping"
            )

        version = data.get(
            "version",
            cls.VERSION,
        )

        if version != cls.VERSION:
            raise HistoryPatternError(
                "unsupported history pattern version: "
                f"{version!r}"
            )

        raw_changes = data.get(
            "changes",
            (),
        )

        if not isinstance(
            raw_changes,
            (list, tuple),
        ):
            raise HistoryPatternError(
                "changes must be a sequence"
            )

        return cls(
            pattern_id=str(data["pattern_id"]),
            changes=tuple(
                IrreversibleChange.from_dict(
                    item
                )
                for item in raw_changes
            ),
            label=data.get("label"),
            description=data.get(
                "description"
            ),
            external_cause_ids=tuple(
                data.get(
                    "external_cause_ids",
                    (),
                )
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )


__all__ = [
    "HistoryPattern",
    "HistoryPatternError",
    "RankedContribution",
]