from experiments.retained_prestress_future_transition_falsification import (
    TOLERANCE,
    run_benchmark,
)


def _data():
    return run_benchmark()


def test_prerequisites_are_valid():
    data = _data()
    assert data["prerequisites_valid"] is True


def test_terminal_connection_states_are_equivalent():
    data = _data()
    terminal = data["terminal_state"]

    assert (
        terminal["connection_state_distance"]
        <= TOLERANCE
    )
    assert terminal["connection_equivalent"] is True


def test_terminal_prestress_states_are_distinguishable():
    data = _data()
    terminal = data["terminal_state"]

    assert terminal["prestress_distance"] > TOLERANCE
    assert terminal["prestress_distinguishable"] is True


def test_registered_read_matches_preregistration():
    data = _data()
    read = data["registered_read_validation"]

    assert read["matches_preregistered_read"] is True
    assert read["event_type"] == "shared_physical_probe"
    assert read["connection_exposure_count"] == 0
    assert read["prestress_perturbation_count"] == 1

    perturbation = read["perturbations"][0]

    assert perturbation["node_id"] == "A"
    assert abs(perturbation["delta"] - 0.30) <= TOLERANCE


def test_identical_read_preserves_connection_equivalence():
    data = _data()
    post = data["post_read"]

    assert post is not None
    assert post["connection_state_distance"] <= TOLERANCE


def test_identical_read_reveals_prestress_divergence():
    data = _data()
    post = data["post_read"]

    assert post is not None
    assert post["prestress_distance"] > TOLERANCE
    assert (
        post["retained_prestress_transition_relevant"]
        is True
    )


def test_registered_decision_is_positive():
    data = _data()

    assert data["decision"] == (
        "RETAINED_PRESTRESS_IS_"
        "TRANSITION_RELEVANT_FOR_REGISTERED_READ"
    )
