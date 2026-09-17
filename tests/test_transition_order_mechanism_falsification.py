"""
Regression tests for the transition-order mechanism falsification benchmark.

These tests freeze the preregistered structural conclusions of:

    experiments.transition_order_mechanism_falsification

They deliberately do NOT freeze the exact observed numerical distance.

Required invariants
-------------------
1. The pure additive reference is order-independent.
2. Fixed-network prestress redistribution is order-independent within
   the registered tolerance for the registered initial state/events.
3. Full reconfiguring evolution is order-dependent in prestress.
4. Reconfiguring evolution produces more AB/BA prestress separation
   than the fixed-network control.
5. Final adaptive connection state is order-independent even though
   final physical prestress state is order-dependent.
6. The benchmark's preregistered decision remains internally
   consistent with those measurements.

Scope
-----
These are computational regression tests only.

They do not establish:
- biological validity,
- clinical validity,
- causal truth in nature,
- global minimality of the full state,
- uniqueness of noncommutativity,
- chaos,
- calibrated probability.
"""

from __future__ import annotations

import pytest

from experiments.transition_order_mechanism_falsification import (
    TOLERANCE,
    run_additive,
    run_benchmark,
    run_fixed_network,
    run_reconfiguring_network,
)


@pytest.fixture(scope="module")
def benchmark():
    """
    Run the complete registered benchmark once for this test module.

    Module scope avoids repeatedly executing the same deterministic
    computational experiment.
    """
    return run_benchmark()


def test_additive_reference_is_order_independent(
    benchmark,
):
    result = benchmark["additive_reference"]

    assert (
        result["prestress_distance"]
        <= TOLERANCE
    )

    assert result["order_dependent"] is False


def test_fixed_network_is_order_independent(
    benchmark,
):
    result = benchmark["fixed_network"]

    assert (
        result["prestress_distance"]
        <= TOLERANCE
    )

    assert result["order_dependent"] is False


def test_reconfiguring_network_is_order_dependent_in_prestress(
    benchmark,
):
    result = benchmark["reconfiguring_network"]

    assert (
        result["prestress_distance"]
        > TOLERANCE
    )

    assert (
        result["order_dependent_prestress"]
        is True
    )


def test_reconfiguration_exceeds_fixed_network_separation(
    benchmark,
):
    fixed_distance = (
        benchmark["fixed_network"][
            "prestress_distance"
        ]
    )

    reconfiguring_distance = (
        benchmark["reconfiguring_network"][
            "prestress_distance"
        ]
    )

    assert (
        reconfiguring_distance
        > fixed_distance + TOLERANCE
    )


def test_final_connection_state_collides_while_prestress_differs(
    benchmark,
):
    """
    Critical structural result.

    AB and BA finish with indistinguishable registered adaptive
    connection state, while their final prestress states remain
    distinguishable.

    Therefore the observed order dependence cannot be attributed
    merely to different final adaptive-connection parameters.
    """

    result = benchmark["reconfiguring_network"]

    assert (
        result["connection_state_distance"]
        <= TOLERANCE
    )

    assert (
        result["prestress_distance"]
        > TOLERANCE
    )

    assert (
        result["order_dependent_connections"]
        is False
    )

    assert (
        result["order_dependent_prestress"]
        is True
    )


def test_registered_state_remains_order_dependent(
    benchmark,
):
    result = benchmark["reconfiguring_network"]

    assert (
        result["registered_state_distance"]
        > TOLERANCE
    )

    assert (
        result["order_dependent_registered_state"]
        is True
    )


def test_additive_ab_and_ba_nodewise_difference_is_zero(
    benchmark,
):
    differences = (
        benchmark["additive_reference"][
            "ab_minus_ba_by_node"
        ]
    )

    assert all(
        abs(float(delta)) <= TOLERANCE
        for delta in differences.values()
    )


def test_fixed_network_ab_and_ba_nodewise_difference_is_zero(
    benchmark,
):
    differences = (
        benchmark["fixed_network"][
            "ab_minus_ba_by_node"
        ]
    )

    assert all(
        abs(float(delta)) <= TOLERANCE
        for delta in differences.values()
    )


def test_reconfiguring_network_contains_real_nodewise_difference(
    benchmark,
):
    differences = (
        benchmark["reconfiguring_network"][
            "ab_minus_ba_by_node"
        ]
    )

    assert any(
        abs(float(delta)) > TOLERANCE
        for delta in differences.values()
    )


def test_preregistered_decision_is_consistent(
    benchmark,
):
    decision = benchmark["decision"]

    assert (
        decision["additive_control_valid"]
        is True
    )

    assert (
        decision["fixed_network_order_dependent"]
        is False
    )

    assert (
        decision[
            "reconfiguring_network_order_dependent"
        ]
        is True
    )

    assert (
        decision["registered_conclusion"]
        ==
        "RECONFIGURATION_INTRODUCES_REGISTERED_ORDER_DEPENDENCE"
    )


def test_claim_scope_remains_bounded(
    benchmark,
):
    assert (
        benchmark["claim_scope"]
        ==
        "computational_registered_system_only"
    )

    excluded = set(
        benchmark["claims_not_tested"]
    )

    required_exclusions = {
        "biological_validity",
        "clinical_validity",
        "causal_truth_in_nature",
        "global_minimality_of_full_state",
        "uniqueness_of_noncommutativity",
        "chaos",
        "calibrated_probability",
    }

    assert required_exclusions <= excluded


def test_direct_helpers_agree_with_complete_benchmark(
    benchmark,
):
    """
    Guard against accidental divergence between the public benchmark
    helpers and run_benchmark().
    """

    additive = run_additive()
    fixed = run_fixed_network()
    reconfiguring = run_reconfiguring_network()

    assert (
        additive["prestress_distance"]
        ==
        pytest.approx(
            benchmark["additive_reference"][
                "prestress_distance"
            ],
            abs=TOLERANCE,
        )
    )

    assert (
        fixed["prestress_distance"]
        ==
        pytest.approx(
            benchmark["fixed_network"][
                "prestress_distance"
            ],
            abs=TOLERANCE,
        )
    )

    assert (
        reconfiguring["prestress_distance"]
        ==
        pytest.approx(
            benchmark["reconfiguring_network"][
                "prestress_distance"
            ],
            abs=TOLERANCE,
        )
    )
