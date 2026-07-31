from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from roif.history import CascadeHistory
from roif.metrics import (
    MetricsError,
    PlaneRanking,
    RankedPlane,
    SimulationMetrics,
    SnapshotMetrics,
    calculate_simulation_metrics,
    calculate_snapshot_metrics,
    event_burden,
    rank_d_fast,
    rank_d_root,
    rank_node_star,
    spectral_coherence,
    spectral_radius,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ranking(
    metric_name: str = "D_fast",
    *,
    scores: tuple[float, ...] = (0.1, 0.2),
    ids: tuple[str, ...] = ("A", "B"),
    higher_is_more_important: bool = False,
) -> PlaneRanking:
    entries = tuple(
        RankedPlane(plane_id=plane_id, score=score, rank=index)
        for index, (plane_id, score) in enumerate(
            zip(ids, scores),
            start=1,
        )
    )
    return PlaneRanking(
        metric_name=metric_name,
        entries=entries,
        higher_is_more_important=higher_is_more_important,
    )


def make_snapshot_metrics(
    *,
    snapshot_id: str = "snapshot-1",
    time: float = 1.0,
    event_count: int = 1,
    burden: float = 2.0,
    maximum_delta: float = 0.5,
) -> SnapshotMetrics:
    return SnapshotMetrics(
        snapshot_id=snapshot_id,
        time=time,
        plane_count=2,
        event_count=event_count,
        history_event_count=event_count,
        changed_plane_count=1,
        maximum_absolute_delta=maximum_delta,
        mean_absolute_delta=maximum_delta / 2.0,
        event_burden=burden,
        d_fast=make_ranking(),
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# spectral_radius
# ---------------------------------------------------------------------------


def test_spectral_radius_identity_matrix() -> None:
    assert spectral_radius([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)


def test_spectral_radius_diagonal_matrix() -> None:
    assert spectral_radius([[2.0, 0.0], [0.0, -3.0]]) == pytest.approx(3.0)


def test_spectral_radius_zero_matrix() -> None:
    assert spectral_radius(np.zeros((3, 3))) == pytest.approx(0.0)


def test_spectral_radius_rotation_matrix_uses_complex_magnitude() -> None:
    matrix = [[0.0, -1.0], [1.0, 0.0]]
    assert spectral_radius(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [1.0, 2.0],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, np.nan], [0.0, 1.0]],
        [[1.0, np.inf], [0.0, 1.0]],
        [["x", 0.0], [0.0, 1.0]],
    ],
)
def test_spectral_radius_rejects_invalid_matrix(matrix) -> None:
    with pytest.raises(MetricsError):
        spectral_radius(matrix)


# ---------------------------------------------------------------------------
# spectral_coherence
# ---------------------------------------------------------------------------


def test_spectral_coherence_positive_below_boundary() -> None:
    eta = spectral_coherence(
        [[0.5, 0.0], [0.0, 0.25]],
        lambda_critical=1.0,
    )
    assert eta == pytest.approx(0.5)


def test_spectral_coherence_zero_at_boundary() -> None:
    eta = spectral_coherence([[2.0]], lambda_critical=2.0)
    assert eta == pytest.approx(0.0)


def test_spectral_coherence_negative_above_boundary() -> None:
    eta = spectral_coherence([[3.0]], lambda_critical=2.0)
    assert eta == pytest.approx(-0.5)


def test_spectral_coherence_clamps_lower_bound() -> None:
    eta = spectral_coherence(
        [[10.0]],
        lambda_critical=1.0,
        clamp=True,
    )
    assert eta == pytest.approx(-1.0)


def test_spectral_coherence_clamps_upper_bound() -> None:
    eta = spectral_coherence(
        [[0.0]],
        lambda_critical=1.0,
        clamp=True,
    )
    assert eta == pytest.approx(1.0)


@pytest.mark.parametrize(
    "lambda_critical",
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, "bad"],
)
def test_spectral_coherence_rejects_invalid_critical_value(
    lambda_critical,
) -> None:
    with pytest.raises(MetricsError):
        spectral_coherence(
            [[1.0]],
            lambda_critical=lambda_critical,
        )


@pytest.mark.parametrize("clamp", [0, 1, "yes", None])
def test_spectral_coherence_requires_boolean_clamp(clamp) -> None:
    with pytest.raises(MetricsError):
        spectral_coherence(
            [[1.0]],
            lambda_critical=1.0,
            clamp=clamp,
        )


# ---------------------------------------------------------------------------
# RankedPlane
# ---------------------------------------------------------------------------


def test_ranked_plane_normalizes_values() -> None:
    item = RankedPlane("  P1  ", "1.25", 1)

    assert item.plane_id == "P1"
    assert item.score == pytest.approx(1.25)
    assert item.rank == 1


def test_ranked_plane_as_dict() -> None:
    item = RankedPlane("P1", 2.5, 3)

    assert item.as_dict() == {
        "plane_id": "P1",
        "score": 2.5,
        "rank": 3,
    }


def test_ranked_plane_is_immutable() -> None:
    item = RankedPlane("P1", 1.0, 1)

    with pytest.raises(FrozenInstanceError):
        item.score = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("plane_id", ["", "   ", 1, None])
def test_ranked_plane_rejects_invalid_plane_id(plane_id) -> None:
    with pytest.raises(MetricsError):
        RankedPlane(plane_id, 1.0, 1)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, "bad"])
def test_ranked_plane_rejects_invalid_score(score) -> None:
    with pytest.raises(MetricsError):
        RankedPlane("P1", score, 1)


@pytest.mark.parametrize("rank", [0, -1, 1.5, True, "1"])
def test_ranked_plane_rejects_invalid_rank(rank) -> None:
    with pytest.raises(MetricsError):
        RankedPlane("P1", 1.0, rank)


# ---------------------------------------------------------------------------
# PlaneRanking
# ---------------------------------------------------------------------------


def test_plane_ranking_accepts_list_and_converts_to_tuple() -> None:
    ranking = PlaneRanking(
        "metric",
        [
            RankedPlane("A", 2.0, 1),
            RankedPlane("B", 1.0, 2),
        ],
    )

    assert isinstance(ranking.entries, tuple)


def test_plane_ranking_winner_and_winner_id() -> None:
    ranking = make_ranking(
        scores=(0.2, 0.8),
        ids=("A", "B"),
        higher_is_more_important=True,
    )

    assert ranking.winner == ranking.entries[0]
    assert ranking.winner_id == "A"


def test_empty_plane_ranking_has_no_winner() -> None:
    ranking = PlaneRanking("empty", tuple())

    assert ranking.winner is None
    assert ranking.winner_id is None
    assert len(ranking) == 0


def test_plane_ranking_is_iterable() -> None:
    ranking = make_ranking()

    assert tuple(ranking) == ranking.entries


def test_plane_ranking_score_for() -> None:
    ranking = make_ranking(scores=(0.1, 0.9))

    assert ranking.score_for(" B ") == pytest.approx(0.9)


def test_plane_ranking_score_for_rejects_unknown_id() -> None:
    ranking = make_ranking()

    with pytest.raises(MetricsError):
        ranking.score_for("X")


@pytest.mark.parametrize("plane_id", ["", "   ", 1, None])
def test_plane_ranking_score_for_rejects_invalid_id(plane_id) -> None:
    ranking = make_ranking()

    with pytest.raises(MetricsError):
        ranking.score_for(plane_id)


def test_plane_ranking_as_dict() -> None:
    ranking = make_ranking()

    payload = ranking.as_dict()

    assert payload["metric_name"] == "D_fast"
    assert payload["higher_is_more_important"] is False
    assert payload["entries"][0]["plane_id"] == "A"


@pytest.mark.parametrize("metric_name", ["", "   ", 1, None])
def test_plane_ranking_rejects_invalid_metric_name(metric_name) -> None:
    with pytest.raises(MetricsError):
        PlaneRanking(metric_name, tuple())


def test_plane_ranking_rejects_non_ranked_entries() -> None:
    with pytest.raises(MetricsError):
        PlaneRanking("metric", ("bad",))


def test_plane_ranking_rejects_duplicate_plane_ids() -> None:
    with pytest.raises(MetricsError):
        PlaneRanking(
            "metric",
            (
                RankedPlane("A", 2.0, 1),
                RankedPlane("A", 1.0, 2),
            ),
        )


def test_plane_ranking_rejects_non_consecutive_ranks() -> None:
    with pytest.raises(MetricsError):
        PlaneRanking(
            "metric",
            (
                RankedPlane("A", 2.0, 1),
                RankedPlane("B", 1.0, 3),
            ),
        )


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_plane_ranking_requires_boolean_direction(value) -> None:
    with pytest.raises(MetricsError):
        PlaneRanking(
            "metric",
            tuple(),
            higher_is_more_important=value,
        )


# ---------------------------------------------------------------------------
# D_fast
# ---------------------------------------------------------------------------


def test_rank_d_fast_places_smallest_reserve_first() -> None:
    ranking = rank_d_fast(
        ["A", "B", "C"],
        [0.4, 0.1, 0.3],
    )

    assert ranking.metric_name == "D_fast"
    assert ranking.higher_is_more_important is False
    assert [entry.plane_id for entry in ranking] == ["B", "C", "A"]
    assert ranking.winner_id == "B"


def test_rank_d_fast_preserves_input_order_for_ties() -> None:
    ranking = rank_d_fast(
        ["A", "B", "C"],
        [0.2, 0.2, 0.2],
    )

    assert [entry.plane_id for entry in ranking] == ["A", "B", "C"]


def test_rank_d_fast_accepts_numpy_array() -> None:
    ranking = rank_d_fast(
        ("A", "B"),
        np.array([0.8, 0.2]),
    )

    assert ranking.winner_id == "B"


@pytest.mark.parametrize(
    ("plane_ids", "reserve"),
    [
        (["A"], []),
        (["A"], [0.1, 0.2]),
        ("A", [0.1]),
        (["A", "A"], [0.1, 0.2]),
        (["A", ""], [0.1, 0.2]),
        (["A", 2], [0.1, 0.2]),
        (["A"], [np.nan]),
        (["A"], [[0.1]]),
    ],
)
def test_rank_d_fast_rejects_invalid_inputs(
    plane_ids,
    reserve,
) -> None:
    with pytest.raises(MetricsError):
        rank_d_fast(plane_ids, reserve)


# ---------------------------------------------------------------------------
# D_root
# ---------------------------------------------------------------------------


def test_rank_d_root_uses_absolute_column_sum() -> None:
    ranking = rank_d_root(
        ["A", "B", "C"],
        [
            [1.0, -4.0, 0.0],
            [2.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
    )

    # Column sums: A=4, B=5, C=2.
    assert [entry.plane_id for entry in ranking] == ["B", "A", "C"]
    assert ranking.score_for("B") == pytest.approx(5.0)
    assert ranking.winner_id == "B"


def test_rank_d_root_preserves_order_for_equal_scores() -> None:
    ranking = rank_d_root(
        ["A", "B"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert [entry.plane_id for entry in ranking] == ["A", "B"]


@pytest.mark.parametrize(
    ("plane_ids", "matrix"),
    [
        (["A"], []),
        (["A"], [1.0]),
        (["A", "B"], [[1.0, 2.0]]),
        (["A"], [[1.0, 0.0], [0.0, 1.0]]),
        (["A", "A"], [[1.0, 0.0], [0.0, 1.0]]),
        (["A"], [[np.inf]]),
    ],
)
def test_rank_d_root_rejects_invalid_inputs(
    plane_ids,
    matrix,
) -> None:
    with pytest.raises(MetricsError):
        rank_d_root(plane_ids, matrix)


# ---------------------------------------------------------------------------
# Node*
# ---------------------------------------------------------------------------


def test_rank_node_star_defaults_to_benefit_score() -> None:
    ranking = rank_node_star(
        ["A", "B", "C"],
        [1.0, 3.0, 2.0],
    )

    assert ranking.metric_name == "Node*"
    assert ranking.winner_id == "B"


def test_rank_node_star_accounts_for_cost() -> None:
    ranking = rank_node_star(
        ["A", "B"],
        [4.0, 6.0],
        cost=[1.0, 3.0],
    )

    assert ranking.score_for("A") == pytest.approx(4.0)
    assert ranking.score_for("B") == pytest.approx(2.0)
    assert ranking.winner_id == "A"


def test_rank_node_star_accounts_for_risk() -> None:
    ranking = rank_node_star(
        ["A", "B"],
        [4.0, 5.0],
        risk=[0.0, 2.0],
        risk_weight=1.0,
    )

    assert ranking.score_for("A") == pytest.approx(4.0)
    assert ranking.score_for("B") == pytest.approx(3.0)
    assert ranking.winner_id == "A"


def test_rank_node_star_combines_benefit_cost_and_risk() -> None:
    ranking = rank_node_star(
        ["A", "B"],
        [8.0, 9.0],
        cost=[2.0, 3.0],
        risk=[0.5, 0.1],
        risk_weight=2.0,
    )

    assert ranking.score_for("A") == pytest.approx(3.0)
    assert ranking.score_for("B") == pytest.approx(2.8)
    assert ranking.winner_id == "A"


def test_rank_node_star_zero_risk_weight_ignores_risk() -> None:
    ranking = rank_node_star(
        ["A", "B"],
        [1.0, 2.0],
        risk=[100.0, 200.0],
        risk_weight=0.0,
    )

    assert ranking.winner_id == "B"


@pytest.mark.parametrize(
    ("plane_ids", "benefit", "cost", "risk", "risk_weight"),
    [
        (["A"], [], None, None, 1.0),
        (["A"], [1.0, 2.0], None, None, 1.0),
        (["A"], [1.0], [1.0, 2.0], None, 1.0),
        (["A"], [1.0], [0.0], None, 1.0),
        (["A"], [1.0], [-1.0], None, 1.0),
        (["A"], [1.0], None, [0.1, 0.2], 1.0),
        (["A"], [1.0], None, None, -1.0),
        (["A"], [1.0], None, None, np.nan),
        (["A"], [1.0], None, None, True),
    ],
)
def test_rank_node_star_rejects_invalid_inputs(
    plane_ids,
    benefit,
    cost,
    risk,
    risk_weight,
) -> None:
    with pytest.raises(MetricsError):
        rank_node_star(
            plane_ids,
            benefit,
            cost=cost,
            risk=risk,
            risk_weight=risk_weight,
        )


# ---------------------------------------------------------------------------
# event_burden
# ---------------------------------------------------------------------------


def test_event_burden_empty_history_is_zero() -> None:
    assert event_burden(CascadeHistory()) == pytest.approx(0.0)


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_event_burden_requires_cascade_history(value) -> None:
    with pytest.raises(MetricsError):
        event_burden(value)


# ---------------------------------------------------------------------------
# SnapshotMetrics
# ---------------------------------------------------------------------------


def test_snapshot_metrics_normalizes_values_and_freezes_metadata() -> None:
    metrics = SnapshotMetrics(
        snapshot_id="  snapshot-1  ",
        time="1.5",
        plane_count=2,
        event_count=1,
        history_event_count=3,
        changed_plane_count=1,
        maximum_absolute_delta="0.8",
        mean_absolute_delta="0.4",
        event_burden="2.0",
        d_fast=make_ranking(),
        metadata={"source": "unit"},
    )

    assert metrics.snapshot_id == "snapshot-1"
    assert metrics.time == pytest.approx(1.5)
    assert isinstance(metrics.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        metrics.metadata["new"] = "value"  # type: ignore[index]


def test_snapshot_metrics_copies_metadata() -> None:
    metadata = {"source": "unit"}
    metrics = make_snapshot_metrics()
    copied = SnapshotMetrics(
        snapshot_id=metrics.snapshot_id,
        time=metrics.time,
        plane_count=metrics.plane_count,
        event_count=metrics.event_count,
        history_event_count=metrics.history_event_count,
        changed_plane_count=metrics.changed_plane_count,
        maximum_absolute_delta=metrics.maximum_absolute_delta,
        mean_absolute_delta=metrics.mean_absolute_delta,
        event_burden=metrics.event_burden,
        d_fast=metrics.d_fast,
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert copied.metadata["source"] == "unit"


def test_snapshot_metrics_as_dict_returns_plain_metadata() -> None:
    metrics = make_snapshot_metrics()

    payload = metrics.as_dict()

    assert payload["snapshot_id"] == "snapshot-1"
    assert payload["metadata"] == {"source": "test"}
    assert isinstance(payload["d_fast"], dict)


def test_snapshot_metrics_is_immutable() -> None:
    metrics = make_snapshot_metrics()

    with pytest.raises(FrozenInstanceError):
        metrics.time = 9.0  # type: ignore[misc]


@pytest.mark.parametrize("snapshot_id", ["", "   ", 1, None])
def test_snapshot_metrics_rejects_invalid_snapshot_id(
    snapshot_id,
) -> None:
    with pytest.raises(MetricsError):
        SnapshotMetrics(
            snapshot_id=snapshot_id,
            time=1.0,
            plane_count=1,
            event_count=0,
            history_event_count=0,
            changed_plane_count=0,
            maximum_absolute_delta=0.0,
            mean_absolute_delta=0.0,
            event_burden=0.0,
            d_fast=make_ranking(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "plane_count",
        "event_count",
        "history_event_count",
        "changed_plane_count",
    ],
)
@pytest.mark.parametrize("bad_value", [-1, 1.5, True, "1"])
def test_snapshot_metrics_rejects_invalid_counts(
    field_name,
    bad_value,
) -> None:
    kwargs = {
        "snapshot_id": "snapshot",
        "time": 1.0,
        "plane_count": 1,
        "event_count": 0,
        "history_event_count": 0,
        "changed_plane_count": 0,
        "maximum_absolute_delta": 0.0,
        "mean_absolute_delta": 0.0,
        "event_burden": 0.0,
        "d_fast": make_ranking(),
    }
    kwargs[field_name] = bad_value

    with pytest.raises(MetricsError):
        SnapshotMetrics(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "time",
        "maximum_absolute_delta",
        "mean_absolute_delta",
        "event_burden",
    ],
)
@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf, -np.inf])
def test_snapshot_metrics_rejects_invalid_non_negative_values(
    field_name,
    bad_value,
) -> None:
    kwargs = {
        "snapshot_id": "snapshot",
        "time": 1.0,
        "plane_count": 1,
        "event_count": 0,
        "history_event_count": 0,
        "changed_plane_count": 0,
        "maximum_absolute_delta": 0.0,
        "mean_absolute_delta": 0.0,
        "event_burden": 0.0,
        "d_fast": make_ranking(),
    }
    kwargs[field_name] = bad_value

    with pytest.raises(MetricsError):
        SnapshotMetrics(**kwargs)


def test_snapshot_metrics_rejects_invalid_d_fast() -> None:
    with pytest.raises(MetricsError):
        SnapshotMetrics(
            snapshot_id="snapshot",
            time=1.0,
            plane_count=1,
            event_count=0,
            history_event_count=0,
            changed_plane_count=0,
            maximum_absolute_delta=0.0,
            mean_absolute_delta=0.0,
            event_burden=0.0,
            d_fast="bad",
        )


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        "bad",
        {1: "value"},
        {"": "value"},
        {"   ": "value"},
    ],
)
def test_snapshot_metrics_rejects_invalid_metadata(metadata) -> None:
    with pytest.raises(MetricsError):
        SnapshotMetrics(
            snapshot_id="snapshot",
            time=1.0,
            plane_count=1,
            event_count=0,
            history_event_count=0,
            changed_plane_count=0,
            maximum_absolute_delta=0.0,
            mean_absolute_delta=0.0,
            event_burden=0.0,
            d_fast=make_ranking(),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# SimulationMetrics
# ---------------------------------------------------------------------------


def test_simulation_metrics_accepts_list_and_converts_to_tuple() -> None:
    metrics = SimulationMetrics(
        [
            make_snapshot_metrics(time=1.0),
            make_snapshot_metrics(
                snapshot_id="snapshot-2",
                time=2.0,
            ),
        ]
    )

    assert isinstance(metrics.snapshots, tuple)
    assert len(metrics) == 2


def test_empty_simulation_metrics_properties() -> None:
    metrics = SimulationMetrics(tuple())

    assert metrics.final is None
    assert metrics.peak_event_burden == pytest.approx(0.0)
    assert metrics.maximum_absolute_delta == pytest.approx(0.0)
    assert metrics.total_current_events == 0


def test_simulation_metrics_aggregate_properties() -> None:
    first = make_snapshot_metrics(
        snapshot_id="s1",
        time=1.0,
        event_count=2,
        burden=3.0,
        maximum_delta=0.4,
    )
    second = make_snapshot_metrics(
        snapshot_id="s2",
        time=2.0,
        event_count=4,
        burden=5.0,
        maximum_delta=0.8,
    )
    metrics = SimulationMetrics((first, second))

    assert metrics.final is second
    assert metrics.peak_event_burden == pytest.approx(5.0)
    assert metrics.maximum_absolute_delta == pytest.approx(0.8)
    assert metrics.total_current_events == 6


def test_simulation_metrics_is_iterable() -> None:
    first = make_snapshot_metrics()
    metrics = SimulationMetrics((first,))

    assert tuple(metrics) == (first,)


def test_simulation_metrics_as_dict() -> None:
    metrics = SimulationMetrics((make_snapshot_metrics(),))

    payload = metrics.as_dict()

    assert payload["snapshot_count"] == 1
    assert payload["peak_event_burden"] == pytest.approx(2.0)
    assert payload["maximum_absolute_delta"] == pytest.approx(0.5)
    assert payload["total_current_events"] == 1
    assert len(payload["snapshots"]) == 1


def test_simulation_metrics_rejects_non_snapshot_metrics() -> None:
    with pytest.raises(MetricsError):
        SimulationMetrics(("bad",))


def test_simulation_metrics_rejects_non_chronological_order() -> None:
    with pytest.raises(MetricsError):
        SimulationMetrics(
            (
                make_snapshot_metrics(
                    snapshot_id="later",
                    time=2.0,
                ),
                make_snapshot_metrics(
                    snapshot_id="earlier",
                    time=1.0,
                ),
            )
        )


def test_simulation_metrics_allows_equal_times() -> None:
    metrics = SimulationMetrics(
        (
            make_snapshot_metrics(
                snapshot_id="s1",
                time=1.0,
            ),
            make_snapshot_metrics(
                snapshot_id="s2",
                time=1.0,
            ),
        )
    )

    assert len(metrics) == 2


def test_simulation_metrics_allows_tiny_float_tolerance() -> None:
    metrics = SimulationMetrics(
        (
            make_snapshot_metrics(
                snapshot_id="s1",
                time=1.0,
            ),
            make_snapshot_metrics(
                snapshot_id="s2",
                time=1.0 - 5e-13,
            ),
        )
    )

    assert len(metrics) == 2


# ---------------------------------------------------------------------------
# Snapshot/simulation integration guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snapshot", [None, object(), {}, []])
def test_calculate_snapshot_metrics_requires_state_snapshot(snapshot) -> None:
    with pytest.raises(MetricsError):
        calculate_snapshot_metrics(snapshot)


@pytest.mark.parametrize("snapshots", ["bad", b"bad", None, [object()]])
def test_calculate_simulation_metrics_rejects_invalid_input(
    snapshots,
) -> None:
    with pytest.raises(MetricsError):
        calculate_simulation_metrics(snapshots)


def test_calculate_simulation_metrics_accepts_empty_sequence() -> None:
    metrics = calculate_simulation_metrics([])

    assert isinstance(metrics, SimulationMetrics)
    assert len(metrics) == 0
    assert metrics.final is None
