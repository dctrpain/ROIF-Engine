"""
Reviewer-facing reproduction runner for Article III.

This module is orchestration only.

It does not implement scientific mechanisms, alter registered benchmark
conditions, or replace the frozen benchmark-specific regression tests.

Scientific freeze:
    Git commit 6dd29ca5fe529f6a285296e08a5e87be98d81f99
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from experiments.transition_order_mechanism_falsification import (
    run_benchmark as run_mechanism_benchmark,
)
from experiments.retained_prestress_future_transition_falsification import (
    run_benchmark as run_retained_prestress_benchmark,
)
from experiments.transition_sufficient_z7_falsification import (
    run_benchmark as run_z7_benchmark,
)
from experiments.transition_sufficient_local_rank_falsification import (
    run_benchmark as run_local_rank_benchmark,
)


SCIENTIFIC_FREEZE_COMMIT = (
    "6dd29ca5fe529f6a285296e08a5e87be98d81f99"
)


@dataclass(frozen=True)
class SuiteResult:
    name: str
    passed: bool
    detail: str


def check_mechanism() -> SuiteResult:
    data = run_mechanism_benchmark()
    decision = data["decision"]

    passed = (
        decision["additive_control_valid"] is True
        and decision["fixed_network_order_dependent"] is False
        and decision["reconfiguring_network_order_dependent"] is True
        and decision["registered_conclusion"]
        == "RECONFIGURATION_INTRODUCES_REGISTERED_ORDER_DEPENDENCE"
    )

    return SuiteResult(
        name="Mechanism decomposition",
        passed=passed,
        detail=decision["registered_conclusion"],
    )


def check_retained_prestress() -> SuiteResult:
    data = run_retained_prestress_benchmark()

    passed = (
        data["prerequisites_valid"] is True
        and data["registered_read_validation"][
            "matches_preregistered_read"
        ]
        is True
        and data["decision"]
        == "RETAINED_PRESTRESS_IS_TRANSITION_RELEVANT_FOR_REGISTERED_READ"
    )

    return SuiteResult(
        name="Retained prestress",
        passed=passed,
        detail=data["decision"],
    )


def check_z7() -> SuiteResult:
    data = run_z7_benchmark()
    decision = data["registered_decision"]

    passed = (
        decision["benchmark_valid"] is True
        and decision["z7_falsified"] is True
        and decision["decision"]
        == "Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN"
    )

    return SuiteResult(
        name="Z7 reduction falsification",
        passed=passed,
        detail=decision["decision"],
    )


def check_local_rank() -> SuiteResult:
    data = run_local_rank_benchmark()
    decision = data["registered_decision"]

    passed = (
        data["deterministic"] is True
        and data["all_finite_difference_states_inside_bounds"] is True
        and data["rank_stable_across_registered_steps"] is True
        and decision["numerical_controls_valid"] is True
        and decision["primary_full_rank"] is True
        and decision["local_lower_bound_supported"] is True
        and decision["decision"]
        == "LOCAL_SMOOTH_EXACT_REDUCTION_BELOW_5D_EXCLUDED"
    )

    return SuiteResult(
        name="Local response-rank audit",
        passed=passed,
        detail=decision["decision"],
    )


def run_regression_tests() -> bool:
    test_files = [
        "tests/test_transition_order_mechanism_falsification.py",
        "tests/test_retained_prestress_future_transition_falsification.py",
        "tests/test_transition_sufficient_z7_falsification.py",
        "tests/test_transition_sufficient_local_rank_falsification.py",
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *test_files,
            "-q",
        ],
        check=False,
    )

    return completed.returncode == 0


def main() -> int:
    print("MANUSCRIPT REPRODUCTION SUITE")
    print("=" * 60)
    print(f"Scientific freeze: {SCIENTIFIC_FREEZE_COMMIT}")
    print()

    checks = [
        check_mechanism(),
        check_retained_prestress(),
        check_z7(),
        check_local_rank(),
    ]

    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name:.<42} {status}")

    print()
    print("Running frozen regression tests...")
    regression_passed = run_regression_tests()
    print(
        "Frozen regression suite"
        f"{'.' * 19} "
        f"{'PASS' if regression_passed else 'FAIL'}"
    )

    overall_passed = (
        all(result.passed for result in checks)
        and regression_passed
    )

    print()
    print("=" * 60)

    if overall_passed:
        print("REGISTERED FALSIFICATION SUITE REPRODUCED")
    else:
        print("REGISTERED FALSIFICATION SUITE FAILED")

    print()
    print("Claim boundaries:")
    print("  Biological validity tested: NO")
    print("  Clinical validity tested: NO")
    print("  Global minimality established: NO")
    print("  All possible reductions excluded: NO")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
