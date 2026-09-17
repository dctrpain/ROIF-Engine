from __future__ import annotations

from experiments.transition_sufficient_z7_falsification import (
    TOLERANCE,
    run_benchmark,
)


def _result():
    return run_benchmark()


def test_registered_dimensions_are_frozen():
    result = _result()

    assert result["registered_state_dimension"] == 13
    assert result["registered_reduction_dimension"] == 7

    assert result["registered_reduction"] == [
        "P_A",
        "P_B",
        "P_C",
        "g_AB",
        "c_AB",
        "g_BC",
        "c_BC",
    ]


def test_source_full_states_are_distinct():
    result = _result()
    initial = result["initial_state"]

    assert initial["full_states_distinct"] is True
    assert initial["full_state_distance"] > TOLERANCE


def test_initial_states_collide_exactly_under_z7():
    result = _result()
    initial = result["initial_state"]

    assert initial["z7_collision"] is True
    assert initial["z7_distance"] <= TOLERANCE
    assert initial["z7_A"] == initial["z7_B"]


def test_direct_mechanical_control_is_equivalent():
    result = _result()
    direct = result["direct_positive_control"]

    assert direct["equivalent"] is True
    assert direct["distance"] <= TOLERANCE


def test_direct_control_preserves_identical_external_response():
    result = _result()
    direct = result["direct_positive_control"]

    assert direct["prestress_A"] == direct["prestress_B"]


def test_identical_write_does_not_directly_change_prestress_difference():
    result = _result()
    write = result["after_identical_write"]

    assert write["prestress_distance"] <= TOLERANCE
    assert write["prestress_A"] == write["prestress_B"]


def test_z7_is_not_closed_under_identical_adaptive_write():
    result = _result()
    write = result["after_identical_write"]

    assert write["z7_closed_under_write"] is False
    assert write["z7_distance"] > TOLERANCE


def test_hidden_decomposition_changes_future_effective_transfer():
    result = _result()
    write = result["after_identical_write"]

    g_a = write["connections_A"]["A_B"]["effective_transfer_gain"]
    g_b = write["connections_B"]["A_B"]["effective_transfer_gain"]

    assert abs(g_a - g_b) > TOLERANCE


def test_common_connection_remains_equivalent():
    result = _result()
    write = result["after_identical_write"]

    bc_a = write["connections_A"]["B_C"]
    bc_b = write["connections_B"]["B_C"]

    assert (
        abs(
            bc_a["effective_transfer_gain"]
            - bc_b["effective_transfer_gain"]
        )
        <= TOLERANCE
    )

    assert (
        abs(
            bc_a["contractile_capacity"]
            - bc_b["contractile_capacity"]
        )
        <= TOLERANCE
    )


def test_identical_future_read_discriminates_collided_sources():
    result = _result()
    future = result["future_identical_read"]

    assert future["discriminates"] is True
    assert future["distance"] > TOLERANCE


def test_negative_repeat_controls_are_zero():
    result = _result()
    controls = result["negative_controls"]

    assert controls["valid"] is True
    assert controls["A_repeat_distance"] <= TOLERANCE
    assert controls["B_repeat_distance"] <= TOLERANCE


def test_registered_decision_falsifies_only_z7():
    result = _result()
    decision = result["registered_decision"]

    assert decision["benchmark_valid"] is True
    assert decision["z7_falsified"] is True
    assert (
        decision["decision"]
        == "Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN"
    )


def test_claim_scope_does_not_overreach():
    result = _result()
    scope = result["claim_scope"]

    assert scope["full_state_globally_minimal_proved"] is False
    assert scope["all_possible_reductions_falsified"] is False
    assert scope["explicit_history_required_proved"] is False
    assert scope["biological_validity_tested"] is False
    assert scope["clinical_validity_tested"] is False
