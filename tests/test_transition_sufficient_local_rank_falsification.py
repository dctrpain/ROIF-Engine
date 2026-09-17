from __future__ import annotations

from experiments.transition_sufficient_local_rank_falsification import (
    run_benchmark,
)


def _result():
    return run_benchmark()


def test_registered_hidden_state_dimension_is_five():
    result = _result()

    assert result["hidden_state_dimension"] == 5
    assert result["hidden_state_names"] == [
        "stiffness",
        "contractile_capacity",
        "reflex_gain",
        "fatigue",
        "remodeling_bias",
    ]


def test_observables_are_only_mechanical_interface_outputs():
    result = _result()

    assert result["observable_per_probe"] == [
        "effective_transfer_gain",
        "contractile_capacity",
    ]


def test_registered_probe_family_is_frozen():
    result = _result()

    assert result["probe_names"] == [
        "zero",
        "load",
        "strain",
        "activation",
        "damage",
        "recovery",
    ]


def test_response_map_is_deterministic():
    result = _result()

    assert result["deterministic"] is True
    assert result["deterministic_repeat_distance"] == 0.0


def test_finite_difference_states_remain_inside_bounds():
    result = _result()

    assert (
        result["all_finite_difference_states_inside_bounds"]
        is True
    )


def test_primary_registered_step_has_full_rank():
    result = _result()

    assert result["primary_finite_difference_step"] == 1e-6
    assert result["primary_rank"] == 5


def test_rank_is_stable_across_registered_steps():
    result = _result()

    assert result["rank_stable_across_registered_steps"] is True

    evaluations = result["evaluations"]

    assert evaluations["1e-05"]["rank"] == 5
    assert evaluations["1e-06"]["rank"] == 5
    assert evaluations["1e-07"]["rank"] == 5


def test_smallest_singular_value_exceeds_registered_threshold():
    result = _result()

    for evaluation in result["evaluations"].values():
        singular_values = evaluation["singular_values"]
        threshold = evaluation["absolute_rank_threshold"]

        assert len(singular_values) == 5
        assert min(singular_values) > threshold


def test_numerical_controls_are_valid():
    result = _result()
    decision = result["registered_decision"]

    assert decision["numerical_controls_valid"] is True


def test_registered_local_lower_bound_is_supported():
    result = _result()
    decision = result["registered_decision"]

    assert decision["primary_full_rank"] is True
    assert decision["local_lower_bound_supported"] is True

    assert (
        decision["decision"]
        == "LOCAL_SMOOTH_EXACT_REDUCTION_BELOW_5D_EXCLUDED"
    )


def test_claim_scope_remains_local_and_restricted():
    result = _result()
    scope = result["claim_scope"]

    assert scope["local_only"] is True
    assert scope["smooth_exact_factorizations_only"] is True

    assert scope["global_minimality_proved"] is False
    assert scope["non_smooth_reductions_excluded"] is False
    assert scope["discontinuous_encodings_excluded"] is False
    assert scope["biological_validity_tested"] is False
    assert scope["clinical_validity_tested"] is False
