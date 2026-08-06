"""
Tests for the universal Probe Graph Adapter.

The adapter converts incomplete graph uncertainty into planning estimates and
Probe evidence into GraphUpdate proposals. It must never apply those proposals
or invoke Solver automatically.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from roif.active_probe_engine import ActiveProbeEngine
from roif.probe_entities import (
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
from roif.probe_graph_adapter import (
    GraphEvidenceDirection,
    GraphEvidenceImpact,
    GraphTarget,
    GraphTargetKind,
    GraphTargetNotFoundError,
    GraphUncertainty,
    GraphUpdateProposal,
    GraphVersionMismatchError,
    IncompleteGraphSnapshot,
    ProbeGraphAdapter,
    ProbeGraphAdapterConfig,
    ProbeGraphAdapterError,
    ProbeGraphLink,
    ProbeGraphLinkNotFoundError,
    ProbeGraphTarget,
)
from roif.probe_planner import ProbeInformationEstimate
from roif.probe_registry import ProbeRegistry


# ============================================================================
# Helpers
# ============================================================================


def make_probe(
    identifier: str = "probe.controlled_load",
) -> ProbeDefinition:
    return ProbeDefinition(
        identifier=identifier,
        name=identifier,
        method=ProbeMethod.INSTRUMENTAL,
        regime=ProbeRegime.STATIC,
        purpose=ProbePurpose.REDUCE_UNCERTAINTY,
        perturbation=Perturbation(
            kind=PerturbationType.LOAD_CHANGE,
            magnitude=1.0,
            magnitude_units="relative",
            duration_seconds=1.0,
        ),
        description="Universal controlled graph Probe.",
        tags=("controlled", "graph"),
    )


def make_node_target(
    identifier: str = "node_A",
) -> GraphTarget:
    return GraphTarget(
        kind=GraphTargetKind.NODE,
        identifier=identifier,
    )


def make_edge_target(
    identifier: str = "edge_A_B",
    *,
    source: str = "node_A",
    target: str = "node_B",
) -> GraphTarget:
    return GraphTarget(
        kind=GraphTargetKind.EDGE,
        identifier=identifier,
        source=source,
        target=target,
    )


def make_uncertainty(
    target: GraphTarget,
    *,
    uncertainty: float = 0.8,
    importance: float = 1.0,
    confidence: float = 0.9,
    hypothesis_count: int = 3,
) -> GraphUncertainty:
    return GraphUncertainty(
        target=target,
        uncertainty=uncertainty,
        importance=importance,
        confidence=confidence,
        hypothesis_count=hypothesis_count,
    )


def make_snapshot(
    *,
    version: str = "v1",
) -> IncompleteGraphSnapshot:
    return IncompleteGraphSnapshot(
        graph_id="graph-1",
        version=version,
        uncertainties=(
            make_uncertainty(
                make_node_target(),
                uncertainty=0.8,
                importance=1.0,
                confidence=0.9,
                hypothesis_count=3,
            ),
            make_uncertainty(
                make_edge_target(),
                uncertainty=0.6,
                importance=0.5,
                confidence=0.8,
                hypothesis_count=2,
            ),
        ),
        metadata={"source": "test"},
    )


def make_link(
    identifier: str = "probe.controlled_load",
    *,
    targets: tuple[ProbeGraphTarget, ...] | None = None,
    novelty: float = 0.6,
    feasibility: float = 0.9,
    estimate_confidence: float = 0.8,
) -> ProbeGraphLink:
    if targets is None:
        targets = (
            ProbeGraphTarget(
                target_identifier="node_A",
                sensitivity=0.9,
                discrimination=0.8,
                expected_uncertainty_reduction=0.7,
            ),
            ProbeGraphTarget(
                target_identifier="edge_A_B",
                sensitivity=0.6,
                discrimination=0.5,
                expected_uncertainty_reduction=0.4,
            ),
        )

    return ProbeGraphLink(
        probe_identifier=identifier,
        targets=targets,
        novelty=novelty,
        feasibility=feasibility,
        estimate_confidence=estimate_confidence,
    )


def make_observation(
    phase: ProbePhase,
    value: object,
    *,
    target: str = "node_A",
    confidence: float = 0.9,
) -> Observation:
    return Observation(
        phase=phase,
        channel=ObservationChannel.DISPLACEMENT,
        target=target,
        value=value,
        units="mm",
        confidence=confidence,
    )


def make_delta(
    *,
    baseline_value: object = 1.0,
    reassessment_value: object = 1.5,
    delta_value: object = 0.5,
    baseline_confidence: float = 0.9,
    reassessment_confidence: float = 0.8,
) -> ObservationDelta:
    return ObservationDelta(
        baseline=make_observation(
            ProbePhase.BASELINE,
            baseline_value,
            confidence=baseline_confidence,
        ),
        reassessment=make_observation(
            ProbePhase.REASSESSMENT,
            reassessment_value,
            confidence=reassessment_confidence,
        ),
        delta=delta_value,
    )


def make_result(
    definition: ProbeDefinition,
    *,
    deltas: tuple[ObservationDelta, ...] = (),
) -> ProbeResult:
    return ProbeResult(
        execution=ProbeExecution(definition=definition),
        deltas=list(deltas),
    )


@pytest.fixture
def probe() -> ProbeDefinition:
    return make_probe()


@pytest.fixture
def snapshot() -> IncompleteGraphSnapshot:
    return make_snapshot()


@pytest.fixture
def link() -> ProbeGraphLink:
    return make_link()


@pytest.fixture
def adapter(
    link: ProbeGraphLink,
) -> ProbeGraphAdapter:
    return ProbeGraphAdapter((link,))


# ============================================================================
# GraphTarget
# ============================================================================


def test_node_target_contract() -> None:
    target = make_node_target()

    assert target.kind is GraphTargetKind.NODE
    assert target.identifier == "node_A"
    assert target.source is None
    assert target.target is None


def test_edge_target_contract() -> None:
    target = make_edge_target()

    assert target.kind is GraphTargetKind.EDGE
    assert target.source == "node_A"
    assert target.target == "node_B"


def test_target_normalizes_identifier() -> None:
    target = GraphTarget(
        kind=GraphTargetKind.NODE,
        identifier="  node_A  ",
    )

    assert target.identifier == "node_A"


def test_target_requires_valid_kind() -> None:
    with pytest.raises(TypeError):
        GraphTarget(
            kind="node",  # type: ignore[arg-type]
            identifier="node_A",
        )


def test_target_requires_nonempty_identifier() -> None:
    with pytest.raises(ValueError):
        GraphTarget(
            kind=GraphTargetKind.NODE,
            identifier=" ",
        )


def test_node_target_rejects_edge_coordinates() -> None:
    with pytest.raises(ValueError):
        GraphTarget(
            kind=GraphTargetKind.NODE,
            identifier="node_A",
            source="node_A",
            target="node_B",
        )


def test_edge_target_requires_source() -> None:
    with pytest.raises(ValueError):
        GraphTarget(
            kind=GraphTargetKind.EDGE,
            identifier="edge_A_B",
            target="node_B",
        )


def test_edge_target_requires_target() -> None:
    with pytest.raises(ValueError):
        GraphTarget(
            kind=GraphTargetKind.EDGE,
            identifier="edge_A_B",
            source="node_A",
        )


def test_target_metadata_is_immutable() -> None:
    target = GraphTarget(
        kind=GraphTargetKind.NODE,
        identifier="node_A",
        metadata={"role": "unknown"},
    )

    assert isinstance(target.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        target.metadata["role"] = "known"  # type: ignore[index]


# ============================================================================
# GraphUncertainty
# ============================================================================


def test_graph_uncertainty_contract() -> None:
    uncertainty = make_uncertainty(make_node_target())

    assert uncertainty.uncertainty == pytest.approx(0.8)
    assert uncertainty.importance == pytest.approx(1.0)
    assert uncertainty.confidence == pytest.approx(0.9)
    assert uncertainty.hypothesis_count == 3


def test_graph_uncertainty_requires_target() -> None:
    with pytest.raises(TypeError):
        GraphUncertainty(
            target=object(),  # type: ignore[arg-type]
            uncertainty=0.5,
        )


@pytest.mark.parametrize(
    "field_name",
    ("uncertainty", "confidence"),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_graph_uncertainty_rejects_invalid_probability(
    field_name: str,
    value: float,
) -> None:
    arguments = {
        "target": make_node_target(),
        "uncertainty": 0.5,
        "confidence": 0.9,
    }

    arguments[field_name] = value

    with pytest.raises(ValueError):
        GraphUncertainty(**arguments)


def test_graph_uncertainty_rejects_negative_importance() -> None:
    with pytest.raises(ValueError):
        GraphUncertainty(
            target=make_node_target(),
            uncertainty=0.5,
            importance=-1.0,
        )


def test_graph_uncertainty_requires_positive_hypothesis_count() -> None:
    with pytest.raises(ValueError):
        GraphUncertainty(
            target=make_node_target(),
            uncertainty=0.5,
            hypothesis_count=0,
        )


def test_graph_uncertainty_rejects_bool_hypothesis_count() -> None:
    with pytest.raises(TypeError):
        GraphUncertainty(
            target=make_node_target(),
            uncertainty=0.5,
            hypothesis_count=True,  # type: ignore[arg-type]
        )


# ============================================================================
# IncompleteGraphSnapshot
# ============================================================================


def test_snapshot_contract(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    assert snapshot.graph_id == "graph-1"
    assert snapshot.version == "v1"
    assert len(snapshot) == 2


def test_snapshot_get_returns_target_uncertainty(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    uncertainty = snapshot.get("node_A")

    assert uncertainty.target.identifier == "node_A"


def test_snapshot_get_normalizes_identifier(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    assert snapshot.get("  node_A  ").target.identifier == "node_A"


def test_snapshot_get_rejects_unknown_target(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(GraphTargetNotFoundError):
        snapshot.get("node_missing")


def test_snapshot_rejects_duplicate_target_identifiers() -> None:
    target = make_node_target()

    with pytest.raises(ValueError):
        IncompleteGraphSnapshot(
            graph_id="graph-1",
            version="v1",
            uncertainties=(
                make_uncertainty(target),
                make_uncertainty(target),
            ),
        )


def test_snapshot_requires_uncertainty_instances() -> None:
    with pytest.raises(TypeError):
        IncompleteGraphSnapshot(
            graph_id="graph-1",
            version="v1",
            uncertainties=(object(),),  # type: ignore[arg-type]
        )


def test_snapshot_total_weight(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    assert snapshot.total_weight == pytest.approx(1.5)


def test_snapshot_total_uncertainty_burden(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    expected = 0.8 * 1.0 + 0.6 * 0.5

    assert snapshot.total_uncertainty_burden == pytest.approx(
        expected
    )


def test_snapshot_metadata_is_immutable(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    assert isinstance(snapshot.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        snapshot.metadata["source"] = "changed"  # type: ignore[index]


# ============================================================================
# ProbeGraphTarget and ProbeGraphLink
# ============================================================================


def test_probe_graph_target_contract() -> None:
    target = ProbeGraphTarget(
        target_identifier="node_A",
        sensitivity=0.8,
        discrimination=0.7,
        expected_uncertainty_reduction=0.6,
    )

    assert target.target_identifier == "node_A"
    assert target.sensitivity == pytest.approx(0.8)


@pytest.mark.parametrize(
    "field_name",
    (
        "sensitivity",
        "discrimination",
        "expected_uncertainty_reduction",
    ),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_probe_graph_target_rejects_invalid_probability(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ProbeGraphTarget(
            target_identifier="node_A",
            **{field_name: value},
        )


def test_probe_graph_link_contract(
    link: ProbeGraphLink,
) -> None:
    assert link.probe_identifier == "probe.controlled_load"
    assert len(link.targets) == 2
    assert link.novelty == pytest.approx(0.6)
    assert link.feasibility == pytest.approx(0.9)


def test_probe_graph_link_rejects_duplicate_targets() -> None:
    target = ProbeGraphTarget(
        target_identifier="node_A"
    )

    with pytest.raises(ValueError):
        ProbeGraphLink(
            probe_identifier="probe.test",
            targets=(target, target),
        )


def test_probe_graph_link_requires_target_instances() -> None:
    with pytest.raises(TypeError):
        ProbeGraphLink(
            probe_identifier="probe.test",
            targets=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    ("novelty", "feasibility", "estimate_confidence"),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_probe_graph_link_rejects_invalid_probability(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ProbeGraphLink(
            probe_identifier="probe.test",
            targets=(),
            **{field_name: value},
        )


# ============================================================================
# Adapter configuration and construction
# ============================================================================


def test_adapter_config_defaults() -> None:
    config = ProbeGraphAdapterConfig()

    assert config.strict_target_resolution
    assert config.require_graph_version_match
    assert config.default_link_novelty == pytest.approx(0.0)
    assert config.maximum_hypothesis_count == 5
    assert config.delta_magnitude_scale == pytest.approx(1.0)


def test_adapter_config_rejects_zero_delta_scale() -> None:
    with pytest.raises(ValueError):
        ProbeGraphAdapterConfig(delta_magnitude_scale=0.0)


def test_adapter_config_requires_at_least_two_hypotheses() -> None:
    with pytest.raises(ValueError):
        ProbeGraphAdapterConfig(maximum_hypothesis_count=1)


def test_adapter_accepts_iterable_links(
    link: ProbeGraphLink,
) -> None:
    adapter = ProbeGraphAdapter((link,))

    assert adapter.get_link(link.probe_identifier) is link


def test_adapter_accepts_mapping_links(
    link: ProbeGraphLink,
) -> None:
    adapter = ProbeGraphAdapter(
        {link.probe_identifier: link}
    )

    assert adapter.get_link(link.probe_identifier) is link


def test_adapter_rejects_mapping_key_mismatch(
    link: ProbeGraphLink,
) -> None:
    with pytest.raises(ProbeGraphAdapterError):
        ProbeGraphAdapter({"probe.wrong": link})


def test_adapter_rejects_duplicate_iterable_links(
    link: ProbeGraphLink,
) -> None:
    with pytest.raises(ProbeGraphAdapterError):
        ProbeGraphAdapter((link, link))


def test_adapter_rejects_non_link_member() -> None:
    with pytest.raises(TypeError):
        ProbeGraphAdapter((object(),))  # type: ignore[arg-type]


def test_adapter_links_mapping_is_immutable(
    adapter: ProbeGraphAdapter,
) -> None:
    links = adapter.links

    assert isinstance(links, MappingProxyType)

    with pytest.raises(TypeError):
        links["probe.new"] = make_link("probe.new")  # type: ignore[index]


def test_get_link_raises_for_unknown_probe(
    adapter: ProbeGraphAdapter,
) -> None:
    with pytest.raises(ProbeGraphLinkNotFoundError):
        adapter.get_link("probe.missing")


# ============================================================================
# Estimate generation
# ============================================================================


def test_estimate_probe_returns_information_estimate(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert isinstance(estimate, ProbeInformationEstimate)
    assert estimate.probe_identifier == probe.identifier


def test_estimate_probe_calculates_positive_information_gain(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.information_gain > 0.0
    assert estimate.information_gain <= 1.0


def test_estimate_probe_calculates_uncertainty_reduction(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.uncertainty_reduction > 0.0
    assert estimate.uncertainty_reduction <= 1.0


def test_full_mapping_has_complete_graph_coverage(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.graph_coverage == pytest.approx(1.0)


def test_partial_mapping_has_partial_graph_coverage(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    link = make_link(
        targets=(
            ProbeGraphTarget(
                target_identifier="node_A"
            ),
        )
    )
    adapter = ProbeGraphAdapter((link,))

    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.graph_coverage == pytest.approx(
        1.0 / 1.5
    )


def test_estimate_preserves_link_novelty_and_feasibility(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.novelty == pytest.approx(0.6)
    assert estimate.feasibility == pytest.approx(0.9)


def test_estimate_confidence_combines_target_and_link_confidence(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    weighted_target_confidence = (
        0.9 * 1.0 + 0.8 * 0.5
    ) / 1.5

    assert estimate.confidence == pytest.approx(
        weighted_target_confidence * 0.8
    )


def test_estimate_metadata_contains_graph_identity(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.metadata["graph_id"] == "graph-1"
    assert estimate.metadata["graph_version"] == "v1"
    assert estimate.metadata["mapping_status"] == "resolved"


def test_missing_link_returns_zero_confidence_estimate(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    probe = make_probe("probe.unmapped")
    adapter = ProbeGraphAdapter(
        config=ProbeGraphAdapterConfig(
            default_link_novelty=0.2
        )
    )

    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.confidence == pytest.approx(0.0)
    assert estimate.information_gain == pytest.approx(0.0)
    assert estimate.novelty == pytest.approx(0.2)
    assert estimate.metadata["mapping_status"] == "missing"


def test_strict_resolution_rejects_unknown_target(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    link = make_link(
        targets=(
            ProbeGraphTarget(
                target_identifier="node_missing"
            ),
        )
    )
    adapter = ProbeGraphAdapter((link,))

    with pytest.raises(GraphTargetNotFoundError):
        adapter.estimate_probe(probe, snapshot)


def test_non_strict_resolution_skips_unknown_target(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    link = make_link(
        targets=(
            ProbeGraphTarget(
                target_identifier="node_missing"
            ),
        )
    )
    adapter = ProbeGraphAdapter(
        (link,),
        config=ProbeGraphAdapterConfig(
            strict_target_resolution=False
        ),
    )

    estimate = adapter.estimate_probe(probe, snapshot)

    assert estimate.confidence == pytest.approx(0.0)
    assert estimate.metadata["mapping_status"] == "empty"


def test_estimate_all_preserves_definition_order(
    snapshot: IncompleteGraphSnapshot,
) -> None:
    probe_b = make_probe("probe.b")
    probe_a = make_probe("probe.a")

    adapter = ProbeGraphAdapter(
        (
            make_link("probe.b"),
            make_link("probe.a"),
        )
    )

    estimates = adapter.estimate_all(
        (probe_b, probe_a),
        snapshot,
    )

    assert tuple(
        estimate.probe_identifier
        for estimate in estimates
    ) == ("probe.b", "probe.a")


def test_estimate_all_accepts_registry(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    estimates = adapter.estimate_all(
        ProbeRegistry((probe,)),
        snapshot,
    )

    assert len(estimates) == 1


def test_estimate_all_accepts_registry_snapshot(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    registry = ProbeRegistry((probe,))

    estimates = adapter.estimate_all(
        registry.snapshot(),
        snapshot,
    )

    assert len(estimates) == 1


def test_estimate_all_rejects_invalid_definition_member(
    adapter: ProbeGraphAdapter,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(TypeError):
        adapter.estimate_all(
            (object(),),  # type: ignore[arg-type]
            snapshot,
        )


# ============================================================================
# Evidence interpretation and GraphUpdateProposal
# ============================================================================


def test_numeric_delta_creates_reducing_impact(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    result = make_result(
        probe,
        deltas=(make_delta(delta_value=0.5),),
    )

    proposal = adapter.propose_graph_update(
        result,
        snapshot,
    )

    impact = proposal.impacts[0]

    assert (
        impact.direction
        is GraphEvidenceDirection.REDUCE_UNCERTAINTY
    )
    assert impact.normalized_magnitude == pytest.approx(0.5)


def test_zero_numeric_delta_creates_neutral_impact(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(make_delta(delta_value=0.0),),
        ),
        snapshot,
    )

    assert (
        proposal.impacts[0].direction
        is GraphEvidenceDirection.NEUTRAL
    )


def test_non_numeric_delta_creates_unknown_impact(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(
                make_delta(
                    baseline_value="before",
                    reassessment_value="after",
                    delta_value={"changed": True},
                ),
            ),
        ),
        snapshot,
    )

    impact = proposal.impacts[0]

    assert impact.direction is GraphEvidenceDirection.UNKNOWN
    assert impact.normalized_magnitude == pytest.approx(0.0)


def test_delta_magnitude_is_scaled_and_capped(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
    link: ProbeGraphLink,
) -> None:
    adapter = ProbeGraphAdapter(
        (link,),
        config=ProbeGraphAdapterConfig(
            delta_magnitude_scale=2.0
        ),
    )

    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(make_delta(delta_value=5.0),),
        ),
        snapshot,
    )

    assert proposal.impacts[0].normalized_magnitude == pytest.approx(
        1.0
    )


def test_impact_confidence_averages_observation_confidence(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(
                make_delta(
                    baseline_confidence=0.8,
                    reassessment_confidence=0.6,
                ),
            ),
        ),
        snapshot,
    )

    assert proposal.impacts[0].confidence == pytest.approx(0.7)


def test_proposal_contains_affected_nodes_and_edges(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(make_delta(),),
        ),
        snapshot,
    )

    assert proposal.graph_update.affected_nodes == ("node_A",)
    assert proposal.graph_update.affected_edges == (
        ("node_A", "node_B"),
    )


def test_proposal_preserves_probe_evidence(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    delta = make_delta()

    proposal = adapter.propose_graph_update(
        make_result(probe, deltas=(delta,)),
        snapshot,
    )

    assert proposal.graph_update.evidence == (delta,)


def test_proposal_is_marked_proposal_only(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(make_delta(),),
        ),
        snapshot,
    )

    assert (
        proposal.graph_update.metadata["application_status"]
        == "proposal_only"
    )
    assert proposal.metadata["automatic_application"] is False


def test_proposal_confidence_is_bounded(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(
            probe,
            deltas=(make_delta(delta_value=1.0),),
        ),
        snapshot,
    )

    assert 0.0 <= proposal.graph_update.confidence <= 1.0


def test_empty_evidence_produces_zero_proposal_confidence(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(probe),
        snapshot,
    )

    assert proposal.graph_update.confidence == pytest.approx(0.0)
    assert proposal.impacts == ()


def test_proposal_validates_graph_version(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(GraphVersionMismatchError):
        adapter.propose_graph_update(
            make_result(probe),
            snapshot,
            graph_version="v2",
        )


def test_matching_graph_version_is_accepted(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(probe),
        snapshot,
        graph_version="v1",
    )

    assert proposal.base_version == "v1"


def test_version_check_can_be_disabled(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
    link: ProbeGraphLink,
) -> None:
    adapter = ProbeGraphAdapter(
        (link,),
        config=ProbeGraphAdapterConfig(
            require_graph_version_match=False
        ),
    )

    proposal = adapter.propose_graph_update(
        make_result(probe),
        snapshot,
        graph_version="v2",
    )

    assert proposal.base_version == "v1"


def test_proposal_requires_registered_link(
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(ProbeGraphLinkNotFoundError):
        ProbeGraphAdapter().propose_graph_update(
            make_result(probe),
            snapshot,
        )


def test_graph_update_proposal_metadata_is_immutable(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    proposal = adapter.propose_graph_update(
        make_result(probe),
        snapshot,
        metadata={"reviewer": "human"},
    )

    assert isinstance(proposal.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        proposal.metadata["reviewer"] = "automatic"  # type: ignore[index]


def test_evidence_impact_metadata_is_immutable() -> None:
    impact = GraphEvidenceImpact(
        delta=make_delta(),
        direction=GraphEvidenceDirection.NEUTRAL,
        normalized_magnitude=0.0,
        confidence=0.9,
        metadata={"source": "test"},
    )

    assert isinstance(impact.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        impact.metadata["source"] = "changed"  # type: ignore[index]


# ============================================================================
# Integration with ActiveProbeEngine
# ============================================================================


def test_engine_result_can_be_converted_to_proposal(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    registry = ProbeRegistry((probe,))
    engine = ActiveProbeEngine(registry)

    engine.start_probe(probe.identifier)
    engine.add_observation(
        make_observation(
            ProbePhase.BASELINE,
            1.0,
        )
    )
    engine.add_observation(
        make_observation(
            ProbePhase.REASSESSMENT,
            1.5,
        )
    )

    result = engine.complete()

    proposal = adapter.propose_graph_update(
        result,
        snapshot,
    )

    assert proposal.probe_identifier == probe.identifier
    assert proposal.graph_update.evidence == tuple(result.deltas)


def test_adapter_estimates_can_be_used_by_engine_planner(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    registry = ProbeRegistry((probe,))
    estimates = adapter.estimate_all(registry, snapshot)

    engine = ActiveProbeEngine(registry)
    plan = engine.plan(estimates)

    assert plan.probe_star is probe


# ============================================================================
# Validation and architectural boundaries
# ============================================================================


def test_estimate_probe_validates_definition(
    adapter: ProbeGraphAdapter,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(TypeError):
        adapter.estimate_probe(
            object(),  # type: ignore[arg-type]
            snapshot,
        )


def test_estimate_probe_validates_snapshot(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
) -> None:
    with pytest.raises(TypeError):
        adapter.estimate_probe(
            probe,
            object(),  # type: ignore[arg-type]
        )


def test_propose_update_validates_result(
    adapter: ProbeGraphAdapter,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    with pytest.raises(TypeError):
        adapter.propose_graph_update(
            object(),  # type: ignore[arg-type]
            snapshot,
        )


def test_propose_update_validates_snapshot(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
) -> None:
    with pytest.raises(TypeError):
        adapter.propose_graph_update(
            make_result(probe),
            object(),  # type: ignore[arg-type]
        )


def test_adapter_has_no_graph_application_method(
    adapter: ProbeGraphAdapter,
) -> None:
    forbidden = (
        "apply_graph_update",
        "apply_update",
        "mutate_graph",
        "commit_graph_update",
    )

    for method_name in forbidden:
        assert not hasattr(adapter, method_name)


def test_adapter_has_no_solver_or_role_methods(
    adapter: ProbeGraphAdapter,
) -> None:
    forbidden = (
        "solve",
        "run_solver",
        "calculate_d_origin",
        "calculate_d_fast",
        "calculate_d_root",
        "calculate_node_star",
    )

    for method_name in forbidden:
        assert not hasattr(adapter, method_name)


def test_estimation_is_deterministic(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    first = adapter.estimate_probe(probe, snapshot)
    second = adapter.estimate_probe(probe, snapshot)

    assert first == second


def test_proposal_generation_is_deterministic(
    adapter: ProbeGraphAdapter,
    probe: ProbeDefinition,
    snapshot: IncompleteGraphSnapshot,
) -> None:
    result = make_result(
        probe,
        deltas=(make_delta(),),
    )

    first = adapter.propose_graph_update(
        result,
        snapshot,
    )
    second = adapter.propose_graph_update(
        result,
        snapshot,
    )

    assert first == second
