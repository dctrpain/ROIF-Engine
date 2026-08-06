"""
Tests for the universal Active Probe Registry.

The registry must remain domain-independent and must not perform planning,
execution, graph mutation, or Solver operations.
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
from roif.probe_registry import (
    DuplicateProbeError,
    ProbeNotFoundError,
    ProbeQuery,
    ProbeRegistry,
    ProbeRegistrySnapshot,
)


# ============================================================================
# Fixtures and helpers
# ============================================================================


def make_probe(
    identifier: str,
    *,
    name: str | None = None,
    method: ProbeMethod = ProbeMethod.OBSERVATION,
    regime: ProbeRegime = ProbeRegime.STATIC,
    purpose: ProbePurpose = ProbePurpose.REDUCE_UNCERTAINTY,
    tags: tuple[str, ...] = (),
    description: str = "",
) -> ProbeDefinition:
    """Create a compact universal ProbeDefinition for registry tests."""

    return ProbeDefinition(
        identifier=identifier,
        name=name or identifier,
        method=method,
        regime=regime,
        purpose=purpose,
        perturbation=Perturbation(
            kind=PerturbationType.NONE,
        ),
        description=description,
        tags=tags,
    )


@pytest.fixture
def observation_probe() -> ProbeDefinition:
    return make_probe(
        "universal.observe",
        name="Universal observation",
        method=ProbeMethod.OBSERVATION,
        regime=ProbeRegime.REST,
        purpose=ProbePurpose.OBSERVE,
        tags=("universal", "passive"),
        description="Observe the system without controlled perturbation.",
    )


@pytest.fixture
def load_probe() -> ProbeDefinition:
    return make_probe(
        "engineering.axial_load",
        name="Axial load experiment",
        method=ProbeMethod.INSTRUMENTAL,
        regime=ProbeRegime.STATIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        tags=("engineering", "load", "structural"),
        description="Controlled axial loading of a structural subsystem.",
    )


@pytest.fixture
def simulation_probe() -> ProbeDefinition:
    return make_probe(
        "universal.virtual_release",
        name="Virtual release",
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.TRANSITION,
        purpose=ProbePurpose.REJECT_HYPOTHESIS,
        tags=("universal", "simulation", "counterfactual"),
        description="Counterfactual simulation of a controlled release.",
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


# ============================================================================
# Construction
# ============================================================================


def test_empty_registry() -> None:
    registry = ProbeRegistry()

    assert len(registry) == 0
    assert registry.identifiers() == ()
    assert registry.definitions() == ()
    assert tuple(registry) == ()


def test_registry_accepts_initial_definitions(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry(
        (observation_probe, load_probe)
    )

    assert len(registry) == 2
    assert registry.identifiers() == (
        observation_probe.identifier,
        load_probe.identifier,
    )


def test_constructor_rejects_duplicate_initial_identifiers(
    observation_probe: ProbeDefinition,
) -> None:
    with pytest.raises(DuplicateProbeError):
        ProbeRegistry(
            (
                observation_probe,
                observation_probe,
            )
        )


def test_constructor_is_atomic_on_invalid_definition(
    observation_probe: ProbeDefinition,
) -> None:
    with pytest.raises(TypeError):
        ProbeRegistry(
            (
                observation_probe,
                object(),
            )
        )


# ============================================================================
# Registration
# ============================================================================


def test_register_adds_definition(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    previous = registry.register(observation_probe)

    assert previous is None
    assert len(registry) == 1
    assert registry.get(observation_probe.identifier) is observation_probe


def test_register_rejects_non_definition() -> None:
    registry = ProbeRegistry()

    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


def test_register_rejects_non_bool_replace(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    with pytest.raises(TypeError):
        registry.register(
            observation_probe,
            replace=1,  # type: ignore[arg-type]
        )


def test_register_rejects_duplicate_identifier(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry((observation_probe,))

    with pytest.raises(DuplicateProbeError):
        registry.register(observation_probe)

    assert registry.get(observation_probe.identifier) is observation_probe


def test_register_replace_returns_previous_definition(
    observation_probe: ProbeDefinition,
) -> None:
    replacement = make_probe(
        observation_probe.identifier,
        name="Replacement",
        method=ProbeMethod.INSTRUMENTAL,
    )

    registry = ProbeRegistry((observation_probe,))

    previous = registry.register(
        replacement,
        replace=True,
    )

    assert previous is observation_probe
    assert registry.get(observation_probe.identifier) is replacement
    assert registry.identifiers() == (
        observation_probe.identifier,
    )


def test_register_many_preserves_input_order(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    previous = registry.register_many(
        (
            observation_probe,
            load_probe,
            simulation_probe,
        )
    )

    assert previous == (None, None, None)
    assert registry.identifiers() == (
        observation_probe.identifier,
        load_probe.identifier,
        simulation_probe.identifier,
    )


def test_register_many_rejects_string_input() -> None:
    registry = ProbeRegistry()

    with pytest.raises(TypeError):
        registry.register_many("not definitions")  # type: ignore[arg-type]


def test_register_many_rejects_non_definition_atomically(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    with pytest.raises(TypeError):
        registry.register_many(
            (
                observation_probe,
                object(),
            )
        )

    assert len(registry) == 0


def test_register_many_rejects_duplicates_inside_batch_atomically(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    with pytest.raises(DuplicateProbeError):
        registry.register_many(
            (
                observation_probe,
                observation_probe,
            )
        )

    assert len(registry) == 0


def test_register_many_rejects_existing_conflict_atomically(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry((observation_probe,))

    new_probe = make_probe("new.probe")

    with pytest.raises(DuplicateProbeError):
        registry.register_many(
            (
                load_probe,
                observation_probe,
                new_probe,
            )
        )

    assert registry.definitions() == (observation_probe,)


def test_register_many_replace_returns_previous_values(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
) -> None:
    replacement = make_probe(
        observation_probe.identifier,
        name="Replacement",
    )

    registry = ProbeRegistry((observation_probe,))

    previous = registry.register_many(
        (
            replacement,
            load_probe,
        ),
        replace=True,
    )

    assert previous == (
        observation_probe,
        None,
    )
    assert registry.definitions() == (
        replacement,
        load_probe,
    )


def test_register_many_rejects_non_bool_replace(
    observation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry()

    with pytest.raises(TypeError):
        registry.register_many(
            (observation_probe,),
            replace="yes",  # type: ignore[arg-type]
        )


# ============================================================================
# Membership and lookup
# ============================================================================


def test_contains_registered_identifier(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    assert observation_probe.identifier in registry


def test_contains_normalizes_outer_whitespace(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    assert f"  {observation_probe.identifier}  " in registry


def test_contains_rejects_non_string_without_error(
    registry: ProbeRegistry,
) -> None:
    assert object() not in registry


def test_get_returns_definition(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    assert registry.get(load_probe.identifier) is load_probe


def test_get_normalizes_identifier_whitespace(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    assert registry.get(
        f"  {load_probe.identifier}  "
    ) is load_probe


def test_get_raises_for_unknown_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ProbeNotFoundError):
        registry.get("missing.probe")


def test_get_rejects_empty_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.get("   ")


def test_get_rejects_non_string_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.get(123)  # type: ignore[arg-type]


def test_find_returns_definition(
    registry: ProbeRegistry,
    simulation_probe: ProbeDefinition,
) -> None:
    assert registry.find(
        simulation_probe.identifier
    ) is simulation_probe


def test_find_returns_none_for_unknown_identifier(
    registry: ProbeRegistry,
) -> None:
    assert registry.find("missing.probe") is None


def test_find_still_validates_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.find("")


# ============================================================================
# Removal
# ============================================================================


def test_remove_returns_removed_definition(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    removed = registry.remove(load_probe.identifier)

    assert removed is load_probe
    assert load_probe.identifier not in registry
    assert len(registry) == 2


def test_remove_raises_for_unknown_identifier(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ProbeNotFoundError):
        registry.remove("missing.probe")


def test_discard_returns_removed_definition(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    removed = registry.discard(
        observation_probe.identifier
    )

    assert removed is observation_probe
    assert observation_probe.identifier not in registry


def test_discard_returns_none_for_unknown_identifier(
    registry: ProbeRegistry,
) -> None:
    assert registry.discard("missing.probe") is None


def test_clear_returns_previous_contents_in_order(
    registry: ProbeRegistry,
) -> None:
    previous = registry.definitions()

    removed = registry.clear()

    assert removed == previous
    assert len(registry) == 0


def test_clear_empty_registry_returns_empty_tuple() -> None:
    registry = ProbeRegistry()

    assert registry.clear() == ()


# ============================================================================
# ProbeQuery validation
# ============================================================================


def test_probe_query_defaults() -> None:
    query = ProbeQuery()

    assert query.method is None
    assert query.regime is None
    assert query.purpose is None
    assert query.required_tags == frozenset()
    assert query.any_tags == frozenset()
    assert query.excluded_tags == frozenset()
    assert query.text is None


def test_probe_query_normalizes_tags() -> None:
    query = ProbeQuery(
        required_tags=frozenset(
            {
                " structural ",
                "load",
            }
        )
    )

    assert query.required_tags == frozenset(
        {
            "structural",
            "load",
        }
    )


def test_probe_query_rejects_required_excluded_overlap() -> None:
    with pytest.raises(ValueError):
        ProbeQuery(
            required_tags=frozenset({"structural"}),
            excluded_tags=frozenset({"structural"}),
        )


def test_probe_query_rejects_wrong_method_type() -> None:
    with pytest.raises(TypeError):
        ProbeQuery(method="instrumental")  # type: ignore[arg-type]


def test_probe_query_rejects_wrong_regime_type() -> None:
    with pytest.raises(TypeError):
        ProbeQuery(regime="static")  # type: ignore[arg-type]


def test_probe_query_rejects_wrong_purpose_type() -> None:
    with pytest.raises(TypeError):
        ProbeQuery(purpose="observe")  # type: ignore[arg-type]


def test_probe_query_rejects_string_as_tag_collection() -> None:
    with pytest.raises(TypeError):
        ProbeQuery(
            required_tags="structural",  # type: ignore[arg-type]
        )


def test_probe_query_rejects_empty_tag() -> None:
    with pytest.raises(ValueError):
        ProbeQuery(
            any_tags=frozenset({" "}),
        )


def test_probe_query_normalizes_empty_text_to_none() -> None:
    query = ProbeQuery(text="   ")

    assert query.text is None


def test_probe_query_rejects_non_string_text() -> None:
    with pytest.raises(TypeError):
        ProbeQuery(text=123)  # type: ignore[arg-type]


# ============================================================================
# Search by taxonomy
# ============================================================================


def test_search_without_filters_returns_all_in_order(
    registry: ProbeRegistry,
) -> None:
    assert registry.search() == registry.definitions()


def test_search_by_method(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        method=ProbeMethod.INSTRUMENTAL,
    )

    assert result == (load_probe,)


def test_search_by_regime(
    registry: ProbeRegistry,
    simulation_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        regime=ProbeRegime.TRANSITION,
    )

    assert result == (simulation_probe,)


def test_search_by_purpose(
    registry: ProbeRegistry,
    simulation_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        purpose=ProbePurpose.REJECT_HYPOTHESIS,
    )

    assert result == (simulation_probe,)


def test_search_combines_taxonomy_filters(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        method=ProbeMethod.INSTRUMENTAL,
        regime=ProbeRegime.STATIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
    )

    assert result == (load_probe,)


def test_search_returns_empty_for_nonmatching_combination(
    registry: ProbeRegistry,
) -> None:
    result = registry.search(
        method=ProbeMethod.SIMULATION,
        regime=ProbeRegime.REST,
    )

    assert result == ()


def test_search_rejects_wrong_method_type(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.search(
            method="instrumental",  # type: ignore[arg-type]
        )


# ============================================================================
# Search by tags
# ============================================================================


def test_required_tags_use_all_semantics(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        required_tags=(
            "engineering",
            "structural",
        )
    )

    assert result == (load_probe,)


def test_required_tags_reject_partial_match(
    registry: ProbeRegistry,
) -> None:
    result = registry.search(
        required_tags=(
            "engineering",
            "simulation",
        )
    )

    assert result == ()


def test_any_tags_use_any_semantics(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        any_tags=(
            "load",
            "simulation",
        )
    )

    assert result == (
        load_probe,
        simulation_probe,
    )


def test_excluded_tags_remove_matching_definitions(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        excluded_tags=("simulation",)
    )

    assert result == (
        observation_probe,
        load_probe,
    )


def test_combined_tag_filters(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        required_tags=("universal",),
        any_tags=(
            "passive",
            "structural",
        ),
        excluded_tags=("simulation",),
    )

    assert result == (observation_probe,)


def test_search_normalizes_tag_whitespace(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(
        required_tags=(" structural ",)
    )

    assert result == (load_probe,)


def test_search_rejects_string_as_tags(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.search(
            required_tags="structural",  # type: ignore[arg-type]
        )


# ============================================================================
# Text search
# ============================================================================


def test_text_search_matches_identifier(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(text="axial_load")

    assert result == (load_probe,)


def test_text_search_matches_name_case_insensitively(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(text="AXIAL LOAD")

    assert result == (load_probe,)


def test_text_search_matches_description(
    registry: ProbeRegistry,
    simulation_probe: ProbeDefinition,
) -> None:
    result = registry.search(text="controlled release")

    assert result == (simulation_probe,)


def test_text_search_matches_tags(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    result = registry.search(text="structural")

    assert result == (load_probe,)


def test_text_search_returns_empty_for_unknown_text(
    registry: ProbeRegistry,
) -> None:
    assert registry.search(text="nonexistent phrase") == ()


def test_empty_text_filter_behaves_as_no_filter(
    registry: ProbeRegistry,
) -> None:
    assert registry.search(text="   ") == registry.definitions()


# ============================================================================
# Query object use
# ============================================================================


def test_search_accepts_probe_query(
    registry: ProbeRegistry,
    load_probe: ProbeDefinition,
) -> None:
    query = ProbeQuery(
        method=ProbeMethod.INSTRUMENTAL,
        required_tags=frozenset({"structural"}),
    )

    assert registry.search(query) == (load_probe,)


def test_search_rejects_query_and_keyword_filters_together(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.search(
            ProbeQuery(),
            method=ProbeMethod.OBSERVATION,
        )


def test_search_rejects_wrong_query_type(
    registry: ProbeRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.search(object())  # type: ignore[arg-type]


# ============================================================================
# Snapshot
# ============================================================================


def test_snapshot_contains_current_registry_contents(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()

    assert isinstance(snapshot, ProbeRegistrySnapshot)
    assert snapshot.definitions == registry.definitions()
    assert len(snapshot) == len(registry)
    assert tuple(snapshot) == registry.definitions()


def test_snapshot_get_returns_definition(
    registry: ProbeRegistry,
    simulation_probe: ProbeDefinition,
) -> None:
    snapshot = registry.snapshot()

    assert snapshot.get(
        simulation_probe.identifier
    ) is simulation_probe


def test_snapshot_get_raises_for_unknown_identifier(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()

    with pytest.raises(ProbeNotFoundError):
        snapshot.get("missing.probe")


def test_snapshot_membership(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    snapshot = registry.snapshot()

    assert observation_probe.identifier in snapshot


def test_snapshot_mapping_is_immutable(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()

    assert isinstance(
        snapshot.by_identifier,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        snapshot.by_identifier["new.probe"] = make_probe(  # type: ignore[index]
            "new.probe"
        )


def test_snapshot_is_independent_from_later_registry_changes(
    registry: ProbeRegistry,
) -> None:
    snapshot = registry.snapshot()
    original_definitions = snapshot.definitions

    registry.clear()
    registry.register(make_probe("new.probe"))

    assert snapshot.definitions == original_definitions
    assert len(snapshot) == 3
    assert "new.probe" not in snapshot


def test_snapshot_rejects_duplicate_definitions(
    observation_probe: ProbeDefinition,
) -> None:
    with pytest.raises(ValueError):
        ProbeRegistrySnapshot(
            definitions=(
                observation_probe,
                observation_probe,
            ),
            by_identifier={
                observation_probe.identifier: observation_probe,
            },
        )


def test_snapshot_rejects_mismatched_mapping(
    observation_probe: ProbeDefinition,
) -> None:
    with pytest.raises(ValueError):
        ProbeRegistrySnapshot(
            definitions=(observation_probe,),
            by_identifier={},
        )


# ============================================================================
# Copy
# ============================================================================


def test_copy_preserves_contents(
    registry: ProbeRegistry,
) -> None:
    copied = registry.copy()

    assert copied is not registry
    assert copied.definitions() == registry.definitions()


def test_copy_is_independent(
    registry: ProbeRegistry,
    observation_probe: ProbeDefinition,
) -> None:
    copied = registry.copy()

    copied.remove(observation_probe.identifier)

    assert observation_probe.identifier not in copied
    assert observation_probe.identifier in registry


# ============================================================================
# Determinism and architectural boundaries
# ============================================================================


def test_iteration_is_deterministic(
    registry: ProbeRegistry,
) -> None:
    first = tuple(registry)
    second = tuple(registry)

    assert first == second
    assert first == registry.definitions()


def test_search_order_follows_registration_order(
    observation_probe: ProbeDefinition,
    load_probe: ProbeDefinition,
    simulation_probe: ProbeDefinition,
) -> None:
    registry = ProbeRegistry(
        (
            simulation_probe,
            observation_probe,
            load_probe,
        )
    )

    result = registry.search(
        any_tags=(
            "universal",
            "engineering",
        )
    )

    assert result == (
        simulation_probe,
        observation_probe,
        load_probe,
    )


def test_registry_stores_definitions_without_execution_state(
    registry: ProbeRegistry,
) -> None:
    for definition in registry:
        assert isinstance(definition, ProbeDefinition)
        assert not hasattr(definition, "baseline_observations")
        assert not hasattr(definition, "reassessment_observations")


def test_registry_has_no_solver_or_planning_methods(
    registry: ProbeRegistry,
) -> None:
    forbidden_methods = (
        "solve",
        "plan",
        "execute",
        "rank",
        "update_graph",
        "apply_graph_update",
    )

    for method_name in forbidden_methods:
        assert not hasattr(registry, method_name)
