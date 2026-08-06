"""
ROIF Engine
===========

Universal registry for Active Probe definitions.

The registry stores reusable ProbeDefinition objects and provides deterministic
lookup and filtering.

Architectural boundaries
------------------------

The registry:

- does not execute Probes;
- does not rank or plan Probes;
- does not mutate system graphs;
- does not calculate uncertainty;
- does not contain domain-specific or medical logic;
- does not integrate directly with Solver.

Planning belongs to probe_policy.py and probe_planner.py.
Execution belongs to active_probe_engine.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .probe_entities import (
    ProbeDefinition,
    ProbeMethod,
    ProbePurpose,
    ProbeRegime,
)


# ============================================================================
# Errors
# ============================================================================


class ProbeRegistryError(Exception):
    """Base exception for Probe Registry failures."""


class DuplicateProbeError(ProbeRegistryError):
    """Raised when a Probe identifier is already registered."""


class ProbeNotFoundError(ProbeRegistryError):
    """Raised when a requested Probe identifier is not registered."""


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_identifier(identifier: str) -> str:
    """Return a normalized non-empty Probe identifier."""

    if not isinstance(identifier, str):
        raise TypeError("identifier must be a string")

    normalized = identifier.strip()

    if not normalized:
        raise ValueError("identifier must not be empty")

    return normalized


def _normalize_tag(tag: str) -> str:
    """Return a normalized non-empty tag."""

    if not isinstance(tag, str):
        raise TypeError("tag must be a string")

    normalized = tag.strip()

    if not normalized:
        raise ValueError("tag must not be empty")

    return normalized


def _normalize_tags(tags: Iterable[str] | None) -> frozenset[str]:
    """Normalize an optional collection of tags."""

    if tags is None:
        return frozenset()

    if isinstance(tags, str):
        raise TypeError("tags must be an iterable of strings, not a string")

    return frozenset(_normalize_tag(tag) for tag in tags)


def _validate_optional_enum(
    value: object,
    expected_type: type,
    field_name: str,
) -> None:
    """Validate an optional Enum filter value."""

    if value is not None and not isinstance(value, expected_type):
        raise TypeError(
            f"{field_name} must be {expected_type.__name__} or None"
        )


# ============================================================================
# Query
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeQuery:
    """
    Immutable filter specification for Probe Registry searches.

    Tag semantics
    -------------

    required_tags:
        Every required tag must be present.

    any_tags:
        At least one of these tags must be present.

    excluded_tags:
        None of these tags may be present.
    """

    method: ProbeMethod | None = None
    regime: ProbeRegime | None = None
    purpose: ProbePurpose | None = None

    required_tags: frozenset[str] = field(default_factory=frozenset)
    any_tags: frozenset[str] = field(default_factory=frozenset)
    excluded_tags: frozenset[str] = field(default_factory=frozenset)

    text: str | None = None

    def __post_init__(self) -> None:
        _validate_optional_enum(
            self.method,
            ProbeMethod,
            "method",
        )
        _validate_optional_enum(
            self.regime,
            ProbeRegime,
            "regime",
        )
        _validate_optional_enum(
            self.purpose,
            ProbePurpose,
            "purpose",
        )

        object.__setattr__(
            self,
            "required_tags",
            _normalize_tags(self.required_tags),
        )
        object.__setattr__(
            self,
            "any_tags",
            _normalize_tags(self.any_tags),
        )
        object.__setattr__(
            self,
            "excluded_tags",
            _normalize_tags(self.excluded_tags),
        )

        overlap = (
            self.required_tags
            & self.excluded_tags
        )

        if overlap:
            raise ValueError(
                "required_tags and excluded_tags overlap: "
                f"{sorted(overlap)!r}"
            )

        if self.text is not None:
            if not isinstance(self.text, str):
                raise TypeError("text must be a string or None")

            normalized_text = self.text.strip()

            object.__setattr__(
                self,
                "text",
                normalized_text or None,
            )


# ============================================================================
# Snapshot
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProbeRegistrySnapshot:
    """
    Immutable point-in-time snapshot of registered Probe definitions.
    """

    definitions: tuple[ProbeDefinition, ...]
    by_identifier: Mapping[str, ProbeDefinition]

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)

        identifiers = tuple(
            definition.identifier
            for definition in definitions
        )

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "snapshot definitions contain duplicate identifiers"
            )

        generated_mapping = {
            definition.identifier: definition
            for definition in definitions
        }

        if dict(self.by_identifier) != generated_mapping:
            raise ValueError(
                "by_identifier does not match definitions"
            )

        object.__setattr__(
            self,
            "definitions",
            definitions,
        )
        object.__setattr__(
            self,
            "by_identifier",
            MappingProxyType(generated_mapping),
        )

    def __len__(self) -> int:
        return len(self.definitions)

    def __iter__(self) -> Iterator[ProbeDefinition]:
        return iter(self.definitions)

    def __contains__(self, identifier: object) -> bool:
        return identifier in self.by_identifier

    def get(self, identifier: str) -> ProbeDefinition:
        """Return one ProbeDefinition from the snapshot."""

        normalized = _validate_identifier(identifier)

        try:
            return self.by_identifier[normalized]
        except KeyError as error:
            raise ProbeNotFoundError(
                f"Probe is not registered: {normalized!r}"
            ) from error


# ============================================================================
# Registry
# ============================================================================


class ProbeRegistry:
    """
    Mutable catalog of reusable ProbeDefinition objects.

    The registry preserves insertion order while exposing deterministic
    filtering and immutable snapshots.
    """

    __slots__ = ("_definitions",)

    def __init__(
        self,
        definitions: Iterable[ProbeDefinition] | None = None,
    ) -> None:
        self._definitions: dict[str, ProbeDefinition] = {}

        if definitions is not None:
            self.register_many(definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    def __iter__(self) -> Iterator[ProbeDefinition]:
        return iter(self._definitions.values())

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str):
            return False

        return identifier.strip() in self._definitions

    def identifiers(self) -> tuple[str, ...]:
        """Return registered identifiers in insertion order."""

        return tuple(self._definitions)

    def definitions(self) -> tuple[ProbeDefinition, ...]:
        """Return registered definitions in insertion order."""

        return tuple(self._definitions.values())

    def register(
        self,
        definition: ProbeDefinition,
        *,
        replace: bool = False,
    ) -> ProbeDefinition | None:
        """
        Register one ProbeDefinition.

        Returns
        -------
        ProbeDefinition | None
            Previous definition when ``replace=True`` and an entry existed;
            otherwise ``None``.
        """

        if not isinstance(definition, ProbeDefinition):
            raise TypeError(
                "definition must be a ProbeDefinition instance"
            )

        if not isinstance(replace, bool):
            raise TypeError("replace must be bool")

        identifier = definition.identifier
        previous = self._definitions.get(identifier)

        if previous is not None and not replace:
            raise DuplicateProbeError(
                f"Probe identifier is already registered: "
                f"{identifier!r}"
            )

        self._definitions[identifier] = definition

        return previous

    def register_many(
        self,
        definitions: Iterable[ProbeDefinition],
        *,
        replace: bool = False,
    ) -> tuple[ProbeDefinition | None, ...]:
        """
        Register multiple definitions atomically.

        If validation fails, the registry remains unchanged.
        """

        if isinstance(definitions, (str, bytes)):
            raise TypeError(
                "definitions must be an iterable of ProbeDefinition objects"
            )

        if not isinstance(replace, bool):
            raise TypeError("replace must be bool")

        incoming = tuple(definitions)

        for definition in incoming:
            if not isinstance(definition, ProbeDefinition):
                raise TypeError(
                    "definitions must contain only ProbeDefinition instances"
                )

        incoming_ids = tuple(
            definition.identifier
            for definition in incoming
        )

        if len(incoming_ids) != len(set(incoming_ids)):
            raise DuplicateProbeError(
                "definitions contain duplicate Probe identifiers"
            )

        if not replace:
            conflicts = sorted(
                identifier
                for identifier in incoming_ids
                if identifier in self._definitions
            )

            if conflicts:
                raise DuplicateProbeError(
                    "Probe identifiers are already registered: "
                    f"{conflicts!r}"
                )

        previous = tuple(
            self._definitions.get(definition.identifier)
            for definition in incoming
        )

        updated = dict(self._definitions)

        for definition in incoming:
            updated[definition.identifier] = definition

        self._definitions = updated

        return previous

    def get(self, identifier: str) -> ProbeDefinition:
        """Return one registered ProbeDefinition."""

        normalized = _validate_identifier(identifier)

        try:
            return self._definitions[normalized]
        except KeyError as error:
            raise ProbeNotFoundError(
                f"Probe is not registered: {normalized!r}"
            ) from error

    def find(self, identifier: str) -> ProbeDefinition | None:
        """
        Return a registered definition or ``None``.

        Unlike ``get()``, this method does not raise ProbeNotFoundError.
        """

        normalized = _validate_identifier(identifier)

        return self._definitions.get(normalized)

    def remove(self, identifier: str) -> ProbeDefinition:
        """Remove and return a registered ProbeDefinition."""

        normalized = _validate_identifier(identifier)

        try:
            return self._definitions.pop(normalized)
        except KeyError as error:
            raise ProbeNotFoundError(
                f"Probe is not registered: {normalized!r}"
            ) from error

    def discard(self, identifier: str) -> ProbeDefinition | None:
        """
        Remove and return a ProbeDefinition, or return ``None``.
        """

        normalized = _validate_identifier(identifier)

        return self._definitions.pop(normalized, None)

    def clear(self) -> tuple[ProbeDefinition, ...]:
        """
        Remove all definitions and return the previous contents.
        """

        previous = self.definitions()
        self._definitions.clear()

        return previous

    def search(
        self,
        query: ProbeQuery | None = None,
        *,
        method: ProbeMethod | None = None,
        regime: ProbeRegime | None = None,
        purpose: ProbePurpose | None = None,
        required_tags: Iterable[str] | None = None,
        any_tags: Iterable[str] | None = None,
        excluded_tags: Iterable[str] | None = None,
        text: str | None = None,
    ) -> tuple[ProbeDefinition, ...]:
        """
        Return definitions matching a ProbeQuery.

        A caller may provide either ``query`` or keyword filters, but not both.
        """

        keyword_filters_used = any(
            value is not None
            for value in (
                method,
                regime,
                purpose,
                required_tags,
                any_tags,
                excluded_tags,
                text,
            )
        )

        if query is not None and keyword_filters_used:
            raise ValueError(
                "provide either query or keyword filters, not both"
            )

        if query is None:
            query = ProbeQuery(
                method=method,
                regime=regime,
                purpose=purpose,
                required_tags=_normalize_tags(required_tags),
                any_tags=_normalize_tags(any_tags),
                excluded_tags=_normalize_tags(excluded_tags),
                text=text,
            )
        elif not isinstance(query, ProbeQuery):
            raise TypeError("query must be ProbeQuery or None")

        return tuple(
            definition
            for definition in self._definitions.values()
            if self._matches(definition, query)
        )

    def snapshot(self) -> ProbeRegistrySnapshot:
        """Return an immutable point-in-time registry snapshot."""

        definitions = self.definitions()

        return ProbeRegistrySnapshot(
            definitions=definitions,
            by_identifier={
                definition.identifier: definition
                for definition in definitions
            },
        )

    def copy(self) -> ProbeRegistry:
        """Return an independent shallow copy of the registry."""

        return ProbeRegistry(self.definitions())

    @staticmethod
    def _matches(
        definition: ProbeDefinition,
        query: ProbeQuery,
    ) -> bool:
        """Return whether one definition matches a query."""

        if (
            query.method is not None
            and definition.method is not query.method
        ):
            return False

        if (
            query.regime is not None
            and definition.regime is not query.regime
        ):
            return False

        if (
            query.purpose is not None
            and definition.purpose is not query.purpose
        ):
            return False

        tags = frozenset(definition.tags)

        if not query.required_tags.issubset(tags):
            return False

        if (
            query.any_tags
            and tags.isdisjoint(query.any_tags)
        ):
            return False

        if tags & query.excluded_tags:
            return False

        if query.text is not None:
            needle = query.text.casefold()

            searchable_text = " ".join(
                (
                    definition.identifier,
                    definition.name,
                    definition.description,
                    " ".join(definition.tags),
                )
            ).casefold()

            if needle not in searchable_text:
                return False

        return True


__all__ = [
    "DuplicateProbeError",
    "ProbeNotFoundError",
    "ProbeQuery",
    "ProbeRegistry",
    "ProbeRegistryError",
    "ProbeRegistrySnapshot",
]
