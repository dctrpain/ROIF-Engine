"""
Tests for roif.history.path_dependence
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roif.history.adaptive_connection import (
    AdaptiveConnectionConfig,
    AdaptiveConnectionState,
    ConnectionExposure,
)
from roif.history.path_dependence import (
    ExposureSequence,
    PathDependenceConfig,
    PathDependenceError,
    PathDependenceResult,
    PathDifference,
    PathExecutionResult,
    PathStep,
    SCHEMA_VERSION,
    compare_exposure_paths,
    compare_sequence_with_reverse,
    execute_exposure_sequence,
    path_dependence_is_policy_free,
    path_difference_vector,
    path_signature,
    reverse_sequence,
)


@pytest.fixture
def source_state():
    return AdaptiveConnectionState(
        connection_id="soft_A_B",
        source_node_id="A",
        target_node_id="B",
        stiffness=2.0,
        contractile_capacity=3.0,
        reflex_gain=1.5,
        fatigue=0.15,
        remodeling_bias=0.0,
        min_stiffness=0.5,
        max_stiffness=5.0,
        min_contractile_capacity=0.5,
        max_contractile_capacity=5.0,
        min_reflex_gain=0.5,
        max_reflex_gain=4.0,
    )


@pytest.fixture
def exposure_a():
    return ConnectionExposure(
        exposure_id="A",
        connection_id="soft_A_B",
        load=0.9,
        strain=0.2,
        activation=0.8,
        damage=0.1,
        recovery=0.0,
    )


@pytest.fixture
def exposure_b():
    return ConnectionExposure(
        exposure_id="B",
        connection_id="soft_A_B",
        load=0.2,
        strain=0.8,
        activation=0.1,
        damage=0.6,
        recovery=0.1,
    )


@pytest.fixture
def sequence_ab(exposure_a, exposure_b):
    return ExposureSequence(
        sequence_id="AB",
        exposures=(
            exposure_a,
            exposure_b,
        ),
    )


@pytest.fixture
def sequence_ba(exposure_a, exposure_b):
    return ExposureSequence(
        sequence_id="BA",
        exposures=(
            exposure_b,
            exposure_a,
        ),
    )


@pytest.fixture
def comparison(
    source_state,
    sequence_ab,
    sequence_ba,
):
    return compare_exposure_paths(
        comparison_id="AB_vs_BA",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
    )


def test_schema_version():
    assert SCHEMA_VERSION == "path_dependence_v1"


def test_sequence_constructs(sequence_ab):
    assert sequence_ab.sequence_id == "AB"
    assert len(sequence_ab.exposures) == 2


def test_empty_sequence_rejected():
    with pytest.raises(PathDependenceError):
        ExposureSequence(
            sequence_id="empty",
            exposures=(),
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_sequence_id_rejected(value, exposure_a):
    with pytest.raises(PathDependenceError):
        ExposureSequence(
            sequence_id=value,
            exposures=(exposure_a,),
        )


def test_negative_tolerance_rejected():
    with pytest.raises(PathDependenceError):
        PathDependenceConfig(
            tolerance=-1e-6,
        )


def test_execute_returns_expected_type(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert isinstance(
        result,
        PathExecutionResult,
    )


def test_execute_preserves_source_state_identity(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert result.source_state == source_state


def test_execute_creates_one_step_per_exposure(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert len(result.steps) == 2


def test_steps_are_path_steps(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert all(
        isinstance(step, PathStep)
        for step in result.steps
    )


def test_step_indices_are_one_based(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert tuple(
        step.step_index
        for step in result.steps
    ) == (1, 2)


def test_step_exposure_ids_follow_sequence(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert tuple(
        step.exposure_id
        for step in result.steps
    ) == ("A", "B")


def test_step_chain_is_contiguous(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert (
        result.steps[0].target_state
        == result.steps[1].source_state
    )


def test_final_state_matches_last_step_target(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="path_ab",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert result.final_state == result.steps[-1].target_state


def test_wrong_connection_in_sequence_rejected(
    source_state,
    exposure_a,
):
    sequence = ExposureSequence(
        sequence_id="bad",
        exposures=(
            exposure_a,
            ConnectionExposure(
                exposure_id="other",
                connection_id="other_connection",
            ),
        ),
    )

    with pytest.raises(PathDependenceError):
        execute_exposure_sequence(
            path_id="bad",
            source_state=source_state,
            sequence=sequence,
        )


def test_reverse_sequence_reverses_exposures(sequence_ab):
    reversed_sequence = reverse_sequence(
        sequence_ab
    )

    assert tuple(
        exposure.exposure_id
        for exposure in reversed_sequence.exposures
    ) == ("B", "A")


def test_reverse_sequence_does_not_mutate_original(sequence_ab):
    before = tuple(
        exposure.exposure_id
        for exposure in sequence_ab.exposures
    )

    reverse_sequence(
        sequence_ab
    )

    after = tuple(
        exposure.exposure_id
        for exposure in sequence_ab.exposures
    )

    assert after == before


def test_reverse_sequence_custom_id(sequence_ab):
    reversed_sequence = reverse_sequence(
        sequence_ab,
        sequence_id="custom_reverse",
    )

    assert reversed_sequence.sequence_id == "custom_reverse"


def test_comparison_returns_expected_type(comparison):
    assert isinstance(
        comparison,
        PathDependenceResult,
    )


def test_difference_type(comparison):
    assert isinstance(
        comparison.difference,
        PathDifference,
    )


def test_ab_vs_ba_is_path_dependent(comparison):
    assert comparison.path_dependent is True


def test_ab_vs_ba_is_not_commutative(comparison):
    assert (
        comparison.commutative_within_tolerance
        is False
    )


def test_path_difference_distance_positive(comparison):
    assert (
        comparison.difference.euclidean_distance
        > 0.0
    )


def test_path_difference_max_component_positive(comparison):
    assert (
        comparison.difference.max_abs_component_delta
        > 0.0
    )


def test_path_difference_vector_has_five_components(comparison):
    vector = path_difference_vector(
        comparison
    )

    assert len(vector) == 5


def test_path_difference_vector_matches_difference(comparison):
    difference = comparison.difference

    assert path_difference_vector(
        comparison
    ) == pytest.approx(
        (
            difference.stiffness_delta,
            difference.contractile_capacity_delta,
            difference.reflex_gain_delta,
            difference.fatigue_delta,
            difference.remodeling_bias_delta,
        )
    )


def test_same_sequence_compares_commutative(
    source_state,
    sequence_ab,
):
    result = compare_exposure_paths(
        comparison_id="same",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ab,
    )

    assert result.path_dependent is False
    assert result.commutative_within_tolerance is True
    assert (
        result.difference.euclidean_distance
        == pytest.approx(0.0)
    )


def test_large_tolerance_can_classify_paths_commutative(
    source_state,
    sequence_ab,
    sequence_ba,
):
    result = compare_exposure_paths(
        comparison_id="large_tol",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
        config=PathDependenceConfig(
            tolerance=1e9,
        ),
    )

    assert result.path_dependent is False
    assert result.commutative_within_tolerance is True


def test_zero_tolerance_preserves_exact_difference(
    source_state,
    sequence_ab,
    sequence_ba,
):
    result = compare_exposure_paths(
        comparison_id="zero_tol",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
        config=PathDependenceConfig(
            tolerance=0.0,
        ),
    )

    assert result.path_dependent is True


def test_compare_sequence_with_reverse_matches_explicit_reverse(
    source_state,
    sequence_ab,
):
    direct = compare_sequence_with_reverse(
        comparison_id="direct",
        source_state=source_state,
        sequence=sequence_ab,
    )

    explicit = compare_exposure_paths(
        comparison_id="explicit",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=reverse_sequence(
            sequence_ab
        ),
    )

    assert (
        direct.difference.euclidean_distance
        == pytest.approx(
            explicit.difference.euclidean_distance
        )
    )


def test_same_events_different_order_preserved(
    comparison,
):
    left_ids = sorted(
        exposure.exposure_id
        for exposure
        in comparison.left_path.sequence.exposures
    )

    right_ids = sorted(
        exposure.exposure_id
        for exposure
        in comparison.right_path.sequence.exposures
    )

    assert left_ids == right_ids == ["A", "B"]


def test_source_state_same_for_both_paths(comparison):
    assert (
        comparison.left_path.source_state
        == comparison.right_path.source_state
    )


def test_final_states_differ_for_path_dependent_case(comparison):
    assert (
        comparison.left_path.final_state
        != comparison.right_path.final_state
    )


def test_custom_adaptive_config_is_supported(
    source_state,
    sequence_ab,
    sequence_ba,
):
    adaptive_config = AdaptiveConnectionConfig(
        fatigue_load_gain=0.2,
        fatigue_strain_gain=0.25,
        fatigue_activation_gain=0.3,
        recovery_gain=0.1,
        damage_capacity_loss_gain=0.7,
        fatigue_capacity_loss_gain=0.4,
        reflex_activation_gain=0.3,
        reflex_damage_gain=0.2,
        stiffness_load_gain=0.25,
        stiffness_damage_gain=0.35,
        remodeling_rate=0.08,
    )

    result = compare_exposure_paths(
        comparison_id="custom",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
        adaptive_config=adaptive_config,
    )

    assert isinstance(
        result,
        PathDependenceResult,
    )


def test_path_metadata_schema(comparison):
    assert (
        comparison.left_path.metadata[
            "schema_version"
        ]
        == SCHEMA_VERSION
    )


def test_result_metadata_schema(comparison):
    assert (
        comparison.metadata[
            "schema_version"
        ]
        == SCHEMA_VERSION
    )


def test_result_declares_no_memory_mutation(comparison):
    assert comparison.metadata[
        "memory_mutated"
    ] is False


def test_result_declares_no_learning(comparison):
    assert comparison.metadata[
        "learning_applied"
    ] is False


def test_result_declares_no_topology_mutation(comparison):
    assert comparison.metadata[
        "topology_modified"
    ] is False


def test_result_declares_no_action(comparison):
    assert comparison.metadata[
        "action_selected"
    ] is False


def test_result_declares_no_policy_modified(comparison):
    assert comparison.metadata[
        "policy_modified"
    ] is False


def test_result_declares_no_diagnosis(comparison):
    assert comparison.metadata[
        "diagnosis_generated"
    ] is False


def test_result_declares_no_biological_truth(comparison):
    assert comparison.metadata[
        "biological_truth_claimed"
    ] is False


def test_result_declares_no_causal_truth(comparison):
    assert comparison.metadata[
        "causal_truth_inferred"
    ] is False


def test_every_step_declares_no_learning(comparison):
    all_steps = (
        comparison.left_path.steps
        + comparison.right_path.steps
    )

    assert all(
        step.metadata[
            "learning_applied"
        ]
        is False
        for step in all_steps
    )


def test_every_step_declares_no_topology_mutation(comparison):
    all_steps = (
        comparison.left_path.steps
        + comparison.right_path.steps
    )

    assert all(
        step.metadata[
            "topology_modified"
        ]
        is False
        for step in all_steps
    )


def test_policy_free_helper(comparison):
    assert path_dependence_is_policy_free(
        comparison
    ) is True


def test_execution_does_not_mutate_source_state(
    source_state,
    sequence_ab,
):
    before = (
        source_state.stiffness,
        source_state.contractile_capacity,
        source_state.reflex_gain,
        source_state.fatigue,
        source_state.remodeling_bias,
    )

    execute_exposure_sequence(
        path_id="immutability",
        source_state=source_state,
        sequence=sequence_ab,
    )

    after = (
        source_state.stiffness,
        source_state.contractile_capacity,
        source_state.reflex_gain,
        source_state.fatigue,
        source_state.remodeling_bias,
    )

    assert after == before


def test_comparison_does_not_mutate_sequences(
    source_state,
    sequence_ab,
    sequence_ba,
):
    left_before = sequence_ab.exposures
    right_before = sequence_ba.exposures

    compare_exposure_paths(
        comparison_id="immutability",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
    )

    assert sequence_ab.exposures == left_before
    assert sequence_ba.exposures == right_before


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExposureSequence(
            sequence_id="s",
            exposures=(
                ConnectionExposure(
                    exposure_id="e",
                    connection_id="c",
                ),
            ),
        ),
        lambda: PathDependenceConfig(),
        lambda: PathDifference(
            stiffness_delta=0.0,
            contractile_capacity_delta=0.0,
            reflex_gain_delta=0.0,
            fatigue_delta=0.0,
            remodeling_bias_delta=0.0,
            euclidean_distance=0.0,
            max_abs_component_delta=0.0,
        ),
    ],
)
def test_primary_objects_are_frozen(factory):
    obj = factory()

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        obj.metadata = {}


def test_path_result_is_frozen(
    source_state,
    sequence_ab,
):
    result = execute_exposure_sequence(
        path_id="frozen",
        source_state=source_state,
        sequence=sequence_ab,
    )

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        result.path_id = "changed"


def test_comparison_result_is_frozen(comparison):
    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        comparison.path_dependent = False


def test_step_is_frozen(comparison):
    step = comparison.left_path.steps[0]

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
            TypeError,
        )
    ):
        step.step_index = 99


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExposureSequence(
            sequence_id="s",
            exposures=(
                ConnectionExposure(
                    exposure_id="e",
                    connection_id="c",
                ),
            ),
            metadata={"x": 1},
        ),
        lambda: PathDependenceConfig(
            metadata={"x": 1},
        ),
        lambda: PathDifference(
            stiffness_delta=0.0,
            contractile_capacity_delta=0.0,
            reflex_gain_delta=0.0,
            fatigue_delta=0.0,
            remodeling_bias_delta=0.0,
            euclidean_distance=0.0,
            max_abs_component_delta=0.0,
            metadata={"x": 1},
        ),
    ],
)
def test_primary_metadata_is_read_only(factory):
    obj = factory()

    with pytest.raises(TypeError):
        obj.metadata["x"] = 2


def test_result_metadata_is_read_only(comparison):
    with pytest.raises(TypeError):
        comparison.metadata[
            "schema_version"
        ] = "changed"


def test_path_metadata_is_read_only(comparison):
    with pytest.raises(TypeError):
        comparison.left_path.metadata[
            "schema_version"
        ] = "changed"


def test_step_metadata_is_read_only(comparison):
    with pytest.raises(TypeError):
        comparison.left_path.steps[
            0
        ].metadata[
            "learning_applied"
        ] = True


def test_difference_metadata_is_read_only(comparison):
    with pytest.raises(TypeError):
        comparison.difference.metadata[
            "difference_direction"
        ] = "changed"


def test_sequence_execution_is_deterministic(
    source_state,
    sequence_ab,
):
    left = execute_exposure_sequence(
        path_id="same",
        source_state=source_state,
        sequence=sequence_ab,
    )

    right = execute_exposure_sequence(
        path_id="same",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert left == right


def test_comparison_is_deterministic(
    source_state,
    sequence_ab,
    sequence_ba,
):
    left = compare_exposure_paths(
        comparison_id="same",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
    )

    right = compare_exposure_paths(
        comparison_id="same",
        source_state=source_state,
        left_sequence=sequence_ab,
        right_sequence=sequence_ba,
    )

    assert left == right


def test_reverse_comparison_is_deterministic(
    source_state,
    sequence_ab,
):
    left = compare_sequence_with_reverse(
        comparison_id="reverse_same",
        source_state=source_state,
        sequence=sequence_ab,
    )

    right = compare_sequence_with_reverse(
        comparison_id="reverse_same",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert left == right


def test_path_signature_is_deterministic(
    source_state,
    sequence_ab,
):
    path = execute_exposure_sequence(
        path_id="signature",
        source_state=source_state,
        sequence=sequence_ab,
    )

    assert path_signature(
        path
    ) == path_signature(
        path
    )


def test_reporting_helpers_are_deterministic(comparison):
    assert path_difference_vector(
        comparison
    ) == path_difference_vector(
        comparison
    )

    assert path_signature(
        comparison.left_path
    ) == path_signature(
        comparison.left_path
    )
