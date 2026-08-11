"""
Tests for roif.hierarchical_memory

The suite fixes the first hierarchical controller-memory contract.

Real integration basis:
- 04B repeated-event no-memory benchmark;
- controller_history_adapter;
- existing StructuralSignature distance geometry.

Core invariants:

    HierarchicalMemory != Learning
    Cluster Assignment != PredictivePreload
    Prototype != Ground Truth
    Compression != Deletion of Anomalies

The suite validates:
- immutable snapshots;
- duplicate protection;
- first-cluster creation;
- nearest-cluster lookup;
- A-family clustering;
- B/C separation from A;
- anomaly handling;
- representatives;
- centroid/prototype updates;
- compression statistics;
- preload-free audit;
- determinism.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from roif.hierarchical_memory import (
    ClusterMatch,
    DuplicateMemoryRecordError,
    HierarchicalMemory,
    HierarchicalMemoryConfig,
    HierarchicalMemoryError,
    InvalidMemoryClusterError,
    MemoryCluster,
    append_controller_history_record,
    append_controller_history_records,
    cluster_by_id,
    hierarchical_memory_is_preload_free,
    memory_compression_stats,
    nearest_cluster,
    record_by_id,
)

from roif.history.controller_history_adapter import (
    adapt_experience_trace,
)

from validation.sailing.sailing_yacht_crew_repeated_event_benchmark import (
    run_repeated_event_benchmark,
)


# =============================================================================
# Fixtures / helpers
# =============================================================================


@pytest.fixture(scope="module")
def benchmark():
    return run_repeated_event_benchmark()


@pytest.fixture(scope="module")
def records(benchmark):
    return tuple(
        adapt_experience_trace(
            episode.trace,
            pattern_id=episode.event.metadata[
                "pattern_family"
            ],
        )
        for episode
        in benchmark.episodes
    )


@pytest.fixture(scope="module")
def default_memory(records):
    return append_controller_history_records(
        HierarchicalMemory(),
        records,
    )


# =============================================================================
# Config
# =============================================================================


def test_default_config_is_valid() -> None:
    config = HierarchicalMemoryConfig()

    assert config.assignment_radius == pytest.approx(
        0.10
    )
    assert config.anomaly_radius == pytest.approx(
        0.50
    )


def test_config_rejects_negative_assignment_radius() -> None:
    with pytest.raises(
        ValueError
    ):
        HierarchicalMemoryConfig(
            assignment_radius=-0.1
        )


def test_config_rejects_anomaly_radius_below_assignment_radius() -> None:
    with pytest.raises(
        HierarchicalMemoryError
    ):
        HierarchicalMemoryConfig(
            assignment_radius=0.5,
            anomaly_radius=0.1,
        )


def test_config_rejects_nonpositive_max_representatives() -> None:
    with pytest.raises(
        HierarchicalMemoryError
    ):
        HierarchicalMemoryConfig(
            max_representatives=0
        )


def test_config_is_frozen() -> None:
    config = HierarchicalMemoryConfig()

    with pytest.raises(
        FrozenInstanceError
    ):
        config.assignment_radius = 1.0  # type: ignore[misc]


# =============================================================================
# Empty memory
# =============================================================================


def test_empty_memory_is_valid() -> None:
    memory = HierarchicalMemory()

    assert memory.record_count == 0
    assert memory.cluster_count == 0


def test_empty_memory_is_preload_free() -> None:
    assert hierarchical_memory_is_preload_free(
        HierarchicalMemory()
    ) is True


def test_nearest_cluster_returns_none_for_empty_memory(
    records,
) -> None:
    assert nearest_cluster(
        HierarchicalMemory(),
        records[
            0
        ],
    ) is None


# =============================================================================
# First append
# =============================================================================


def test_first_append_creates_new_snapshot(
    records,
) -> None:
    base = HierarchicalMemory()

    updated = append_controller_history_record(
        base,
        records[
            0
        ],
    )

    assert updated is not base
    assert base.record_count == 0
    assert updated.record_count == 1


def test_first_append_creates_one_cluster(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    assert memory.cluster_count == 1


def test_first_cluster_contains_first_record(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    cluster = memory.clusters[
        0
    ]

    assert cluster.member_record_ids == (
        records[
            0
        ].trace_id,
    )


def test_first_cluster_radius_is_zero(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    assert memory.clusters[
        0
    ].local_radius == pytest.approx(
        0.0
    )


def test_first_cluster_representative_is_first_record(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    assert memory.clusters[
        0
    ].representative_record_ids == (
        records[
            0
        ].trace_id,
    )


# =============================================================================
# Duplicate protection
# =============================================================================


def test_duplicate_record_is_rejected(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    with pytest.raises(
        DuplicateMemoryRecordError
    ):
        append_controller_history_record(
            memory,
            records[
                0
            ],
        )


def test_constructor_rejects_duplicate_records(
    records,
) -> None:
    with pytest.raises(
        DuplicateMemoryRecordError
    ):
        HierarchicalMemory(
            records=(
                records[
                    0
                ],
                records[
                    0
                ],
            )
        )


# =============================================================================
# A-family clustering
# =============================================================================


def test_second_a_is_within_assignment_radius(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    match = nearest_cluster(
        memory,
        records[
            2
        ],
    )

    assert isinstance(
        match,
        ClusterMatch,
    )

    assert (
        match.within_assignment_radius
        is True
    )


def test_second_a_joins_first_cluster(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
        ),
    )

    assert memory.cluster_count == 1

    assert memory.clusters[
        0
    ].size == 2


def test_third_a_joins_same_cluster(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
            records[
                4
            ],
        ),
    )

    assert memory.cluster_count == 1
    assert memory.clusters[
        0
    ].size == 3


def test_a_cluster_contains_exact_a_trace_ids(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
            records[
                4
            ],
        ),
    )

    assert memory.clusters[
        0
    ].member_record_ids == (
        records[
            0
        ].trace_id,
        records[
            2
        ].trace_id,
        records[
            4
        ].trace_id,
    )


def test_a_cluster_has_no_anomalies(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
            records[
                4
            ],
        ),
    )

    assert memory.clusters[
        0
    ].anomaly_record_ids == ()


# =============================================================================
# Cross-family structure
# =============================================================================


def test_b_is_outside_a_anomaly_radius(
    records,
) -> None:
    memory = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    match = nearest_cluster(
        memory,
        records[
            1
        ],
    )

    assert match is not None
    assert (
        match.within_anomaly_radius
        is False
    )


def test_b_creates_second_cluster(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                1
            ],
        ),
    )

    assert memory.cluster_count == 2


def test_c_is_close_enough_to_b_cluster_under_default_radius(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                1
            ],
        ),
    )

    match = nearest_cluster(
        memory,
        records[
            3
        ],
    )

    assert match is not None

    # B-C distance ~= 0.117: outside assignment radius 0.10,
    # but inside anomaly radius 0.50.
    assert (
        match.within_assignment_radius
        is False
    )
    assert (
        match.within_anomaly_radius
        is True
    )


def test_c_is_preserved_as_anomaly_in_b_cluster(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                1
            ],
            records[
                3
            ],
        ),
    )

    assert memory.cluster_count == 2

    clusters_by_pattern = {
        tuple(
            cluster.member_record_ids
        ): cluster
        for cluster
        in memory.clusters
    }

    bc_cluster = next(
        cluster
        for cluster
        in memory.clusters
        if records[
            1
        ].trace_id
        in cluster.member_record_ids
    )

    assert records[
        3
    ].trace_id in bc_cluster.member_record_ids

    assert records[
        3
    ].trace_id in bc_cluster.anomaly_record_ids


def test_default_full_memory_has_two_clusters(
    default_memory,
) -> None:
    assert default_memory.cluster_count == 2


def test_default_full_memory_preserves_five_records(
    default_memory,
) -> None:
    assert default_memory.record_count == 5


# =============================================================================
# Record and cluster lookup
# =============================================================================


def test_record_by_id_returns_record(
    default_memory,
    records,
) -> None:
    result = record_by_id(
        default_memory,
        records[
            0
        ].trace_id,
    )

    assert result == records[
        0
    ]


def test_record_by_id_rejects_unknown_id(
    default_memory,
) -> None:
    with pytest.raises(
        KeyError
    ):
        record_by_id(
            default_memory,
            "unknown",
        )


def test_cluster_by_id_returns_cluster(
    default_memory,
) -> None:
    cluster = default_memory.clusters[
        0
    ]

    assert cluster_by_id(
        default_memory,
        cluster.cluster_id,
    ) == cluster


def test_cluster_by_id_rejects_unknown_id(
    default_memory,
) -> None:
    with pytest.raises(
        KeyError
    ):
        cluster_by_id(
            default_memory,
            "unknown",
        )


# =============================================================================
# Cluster invariants
# =============================================================================


def test_cluster_rejects_empty_id(
    records,
) -> None:
    vector = tuple(
        records[
            0
        ].signature.compact_vector()
    )

    with pytest.raises(
        InvalidMemoryClusterError
    ):
        MemoryCluster(
            cluster_id="",
            member_record_ids=(
                records[
                    0
                ].trace_id,
            ),
            prototype_vector=vector,
            local_radius=0.0,
            representative_record_ids=(
                records[
                    0
                ].trace_id,
            ),
        )


def test_cluster_rejects_empty_members(
    records,
) -> None:
    vector = tuple(
        records[
            0
        ].signature.compact_vector()
    )

    with pytest.raises(
        InvalidMemoryClusterError
    ):
        MemoryCluster(
            cluster_id="x",
            member_record_ids=(),
            prototype_vector=vector,
            local_radius=0.0,
            representative_record_ids=(),
        )


def test_cluster_rejects_representative_not_in_members(
    records,
) -> None:
    vector = tuple(
        records[
            0
        ].signature.compact_vector()
    )

    with pytest.raises(
        InvalidMemoryClusterError
    ):
        MemoryCluster(
            cluster_id="x",
            member_record_ids=(
                records[
                    0
                ].trace_id,
            ),
            prototype_vector=vector,
            local_radius=0.0,
            representative_record_ids=(
                "other",
            ),
        )


def test_cluster_rejects_anomaly_not_in_members(
    records,
) -> None:
    vector = tuple(
        records[
            0
        ].signature.compact_vector()
    )

    with pytest.raises(
        InvalidMemoryClusterError
    ):
        MemoryCluster(
            cluster_id="x",
            member_record_ids=(
                records[
                    0
                ].trace_id,
            ),
            prototype_vector=vector,
            local_radius=0.0,
            representative_record_ids=(
                records[
                    0
                ].trace_id,
            ),
            anomaly_record_ids=(
                "other",
            ),
        )


# =============================================================================
# Prototype / representatives
# =============================================================================


def test_a_cluster_prototype_has_same_vector_length(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
            records[
                4
            ],
        ),
    )

    assert len(
        memory.clusters[
            0
        ].prototype_vector
    ) == len(
        records[
            0
        ].signature.compact_vector()
    )


def test_a_cluster_radius_is_positive_after_multiple_members(
    records,
) -> None:
    memory = append_controller_history_records(
        HierarchicalMemory(),
        (
            records[
                0
            ],
            records[
                2
            ],
        ),
    )

    assert memory.clusters[
        0
    ].local_radius > 0.0


def test_representatives_are_cluster_members(
    default_memory,
) -> None:
    for cluster in default_memory.clusters:
        assert set(
            cluster.representative_record_ids
        ).issubset(
            set(
                cluster.member_record_ids
            )
        )


def test_default_representative_count_never_exceeds_config(
    default_memory,
) -> None:
    for cluster in default_memory.clusters:
        assert len(
            cluster.representative_record_ids
        ) <= default_memory.config.max_representatives


# =============================================================================
# Compression stats
# =============================================================================


def test_compression_stats_preserve_raw_count(
    default_memory,
) -> None:
    stats = memory_compression_stats(
        default_memory
    )

    assert stats.raw_record_count == 5


def test_compression_stats_preserve_cluster_count(
    default_memory,
) -> None:
    stats = memory_compression_stats(
        default_memory
    )

    assert stats.cluster_count == 2


def test_compression_stats_have_positive_ratio(
    default_memory,
) -> None:
    stats = memory_compression_stats(
        default_memory
    )

    assert stats.compression_ratio > 0.0


def test_empty_memory_compression_ratio_is_one() -> None:
    stats = memory_compression_stats(
        HierarchicalMemory()
    )

    assert stats.compression_ratio == pytest.approx(
        1.0
    )


# =============================================================================
# Audit boundary
# =============================================================================


def test_default_memory_is_preload_free(
    default_memory,
) -> None:
    assert hierarchical_memory_is_preload_free(
        default_memory
    ) is True


def test_default_memory_marks_learning_disabled(
    default_memory,
) -> None:
    assert (
        default_memory.metadata[
            "learning_applied"
        ]
        is False
    )


def test_default_memory_marks_predictive_preload_disabled(
    default_memory,
) -> None:
    assert (
        default_memory.metadata[
            "predictive_preload_applied"
        ]
        is False
    )


def test_default_memory_marks_policy_unmodified(
    default_memory,
) -> None:
    assert (
        default_memory.metadata[
            "policy_modified"
        ]
        is False
    )


def test_default_memory_marks_graph_unmutated(
    default_memory,
) -> None:
    assert (
        default_memory.metadata[
            "graph_mutated"
        ]
        is False
    )


def test_preload_free_detects_forbidden_memory_flag(
    default_memory,
) -> None:
    altered = replace(
        default_memory,
        metadata={
            **dict(
                default_memory.metadata
            ),
            "predictive_preload_applied": True,
        },
    )

    assert hierarchical_memory_is_preload_free(
        altered
    ) is False


# =============================================================================
# Immutability
# =============================================================================


def test_memory_records_are_tuple(
    default_memory,
) -> None:
    assert isinstance(
        default_memory.records,
        tuple,
    )


def test_memory_clusters_are_tuple(
    default_memory,
) -> None:
    assert isinstance(
        default_memory.clusters,
        tuple,
    )


def test_memory_metadata_is_read_only(
    default_memory,
) -> None:
    assert isinstance(
        default_memory.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        default_memory.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_memory_is_frozen(
    default_memory,
) -> None:
    with pytest.raises(
        FrozenInstanceError
    ):
        default_memory.records = ()  # type: ignore[misc]


def test_cluster_metadata_is_read_only(
    default_memory,
) -> None:
    cluster = default_memory.clusters[
        0
    ]

    assert isinstance(
        cluster.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        cluster.metadata[
            "x"
        ] = 1  # type: ignore[index]


def test_cluster_is_frozen(
    default_memory,
) -> None:
    cluster = default_memory.clusters[
        0
    ]

    with pytest.raises(
        FrozenInstanceError
    ):
        cluster.local_radius = 1.0  # type: ignore[misc]


# =============================================================================
# Determinism
# =============================================================================


def test_single_append_is_deterministic(
    records,
) -> None:
    left = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    right = append_controller_history_record(
        HierarchicalMemory(),
        records[
            0
        ],
    )

    assert left == right


def test_a_cluster_construction_is_deterministic(
    records,
) -> None:
    sequence = (
        records[
            0
        ],
        records[
            2
        ],
        records[
            4
        ],
    )

    left = append_controller_history_records(
        HierarchicalMemory(),
        sequence,
    )

    right = append_controller_history_records(
        HierarchicalMemory(),
        sequence,
    )

    assert left == right


def test_full_04b_hierarchical_memory_is_deterministic(
    records,
) -> None:
    left = append_controller_history_records(
        HierarchicalMemory(),
        records,
    )

    right = append_controller_history_records(
        HierarchicalMemory(),
        records,
    )

    assert left == right


def test_full_04b_cluster_memberships_are_deterministic(
    records,
) -> None:
    left = append_controller_history_records(
        HierarchicalMemory(),
        records,
    )

    right = append_controller_history_records(
        HierarchicalMemory(),
        records,
    )

    assert tuple(
        cluster.member_record_ids
        for cluster
        in left.clusters
    ) == tuple(
        cluster.member_record_ids
        for cluster
        in right.clusters
    )
