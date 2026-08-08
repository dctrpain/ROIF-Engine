"""
Mechanical Validation — Scenario 02P
Physics Ground Truth for a Pre-Stressed Parallel Spring Network

This module is deliberately independent of ROIF.

Its purpose is to define a small mechanical benchmark whose behavior can be
computed analytically from ordinary linear axial mechanics before ROIF sees
the system.

Physical layout
---------------

                    ┌── Spring A (primary) ──┐
ANCHOR / LOAD  ─────┤                       ├──── JUNCTION ─── Spring C ─── OUTPUT
                    └── Spring B (bypass) ──┘

A and B are parallel axial members.
C is a downstream series member.

The total tensile load is force-controlled:

    F_total = preload + external_increment

For two parallel linear springs under the same elongation:

    F_A = k_A / (k_A + k_B) * F_total
    F_B = k_B / (k_A + k_B) * F_total

with:

    k = E * A / L

Scenario
--------
Healthy state:
    A is stiffer than B and both remain below their allowable tensile force.

Defect:
    A suffers a large effective stiffness loss while the total tensile load
    remains unchanged.

Consequence:
    load redistributes toward B;
    B exceeds its allowable tensile force first;
    if B is then lost, A must carry the full force and also exceeds its limit.

This gives an analytically defined cascade without any biological semantics
and without using ROIF to generate the expected behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from types import MappingProxyType
from typing import Mapping


class MechanicalGroundTruthError(ValueError):
    """Raised when the analytical benchmark violates its physical contract."""


@dataclass(frozen=True, slots=True)
class AxialMember:
    """
    Linear axial tension member.

    Parameters use SI units:
        length_m            m
        area_m2             m^2
        young_modulus_pa    Pa
        allowable_force_n   N
    """

    member_id: str
    length_m: float
    area_m2: float
    young_modulus_pa: float
    allowable_force_n: float

    def __post_init__(self) -> None:
        if not self.member_id:
            raise MechanicalGroundTruthError("member_id must not be empty.")

        for name, value in (
            ("length_m", self.length_m),
            ("area_m2", self.area_m2),
            ("young_modulus_pa", self.young_modulus_pa),
            ("allowable_force_n", self.allowable_force_n),
        ):
            if value <= 0.0:
                raise MechanicalGroundTruthError(
                    f"{name} must be positive."
                )

    @property
    def axial_stiffness_n_per_m(self) -> float:
        return (
            self.young_modulus_pa
            * self.area_m2
            / self.length_m
        )

    def force_utilization(self, force_n: float) -> float:
        return abs(float(force_n)) / self.allowable_force_n

    def is_overloaded(self, force_n: float) -> bool:
        return self.force_utilization(force_n) > 1.0


@dataclass(frozen=True, slots=True)
class LoadCase:
    preload_n: float
    external_increment_n: float

    @property
    def total_force_n(self) -> float:
        return self.preload_n + self.external_increment_n


@dataclass(frozen=True, slots=True)
class ParallelForceState:
    total_force_n: float
    elongation_m: float
    primary_force_n: float
    bypass_force_n: float
    downstream_force_n: float
    primary_utilization: float
    bypass_utilization: float
    downstream_utilization: float
    primary_overloaded: bool
    bypass_overloaded: bool
    downstream_overloaded: bool

    @property
    def any_overload(self) -> bool:
        return (
            self.primary_overloaded
            or self.bypass_overloaded
            or self.downstream_overloaded
        )


@dataclass(frozen=True, slots=True)
class FailureStep:
    step_index: int
    failed_member_id: str
    force_at_failure_n: float
    utilization: float
    surviving_member_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MechanicalGroundTruth:
    case_id: str
    healthy_primary: AxialMember
    defective_primary: AxialMember
    bypass: AxialMember
    downstream: AxialMember
    load_case: LoadCase
    healthy_state: ParallelForceState
    defective_state: ParallelForceState
    failure_sequence: tuple[FailureStep, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failure_sequence",
            tuple(self.failure_sequence),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def parallel_force_state(
    primary: AxialMember,
    bypass: AxialMember,
    downstream: AxialMember,
    *,
    total_force_n: float,
) -> ParallelForceState:
    if total_force_n < 0.0:
        raise MechanicalGroundTruthError(
            "This benchmark is tension-only; total force must be non-negative."
        )

    k_primary = primary.axial_stiffness_n_per_m
    k_bypass = bypass.axial_stiffness_n_per_m
    k_parallel = k_primary + k_bypass

    if k_parallel <= 0.0:
        raise MechanicalGroundTruthError(
            "Equivalent parallel stiffness must be positive."
        )

    elongation_m = total_force_n / k_parallel

    primary_force_n = k_primary * elongation_m
    bypass_force_n = k_bypass * elongation_m
    downstream_force_n = total_force_n

    return ParallelForceState(
        total_force_n=total_force_n,
        elongation_m=elongation_m,
        primary_force_n=primary_force_n,
        bypass_force_n=bypass_force_n,
        downstream_force_n=downstream_force_n,
        primary_utilization=primary.force_utilization(primary_force_n),
        bypass_utilization=bypass.force_utilization(bypass_force_n),
        downstream_utilization=downstream.force_utilization(
            downstream_force_n
        ),
        primary_overloaded=primary.is_overloaded(primary_force_n),
        bypass_overloaded=bypass.is_overloaded(bypass_force_n),
        downstream_overloaded=downstream.is_overloaded(
            downstream_force_n
        ),
    )


def force_after_single_branch_loss(
    surviving_member: AxialMember,
    *,
    total_force_n: float,
) -> tuple[float, float, bool]:
    force_n = total_force_n
    utilization = surviving_member.force_utilization(force_n)
    return (
        force_n,
        utilization,
        surviving_member.is_overloaded(force_n),
    )


def build_physics_ground_truth() -> MechanicalGroundTruth:
    """
    Deterministic synthetic benchmark in SI units.

    Geometry:
        L = 1.0 m
        A = 1e-4 m² = 100 mm²

    Therefore:
        E = 10 MPa  -> k = 1000 N/m
        E = 5 MPa   -> k = 500 N/m
        E = 2.5 MPa -> k = 250 N/m
        E = 8 MPa   -> k = 800 N/m

    These are synthetic linear-elastic members, not claims about a named
    commercial material.
    """

    healthy_primary = AxialMember(
        member_id="primary_branch",
        length_m=1.0,
        area_m2=1.0e-4,
        young_modulus_pa=10.0e6,
        allowable_force_n=190.0,
    )

    defective_primary = AxialMember(
        member_id="primary_branch",
        length_m=1.0,
        area_m2=1.0e-4,
        young_modulus_pa=2.5e6,
        allowable_force_n=190.0,
    )

    bypass = AxialMember(
        member_id="bypass_branch",
        length_m=1.0,
        area_m2=1.0e-4,
        young_modulus_pa=5.0e6,
        allowable_force_n=130.0,
    )

    downstream = AxialMember(
        member_id="downstream_member",
        length_m=1.0,
        area_m2=1.0e-4,
        young_modulus_pa=8.0e6,
        allowable_force_n=300.0,
    )

    load_case = LoadCase(
        preload_n=120.0,
        external_increment_n=120.0,
    )

    total_force_n = load_case.total_force_n

    healthy_state = parallel_force_state(
        healthy_primary,
        bypass,
        downstream,
        total_force_n=total_force_n,
    )

    defective_state = parallel_force_state(
        defective_primary,
        bypass,
        downstream,
        total_force_n=total_force_n,
    )

    if defective_state.bypass_overloaded is not True:
        raise MechanicalGroundTruthError(
            "Benchmark design error: bypass must overload after primary stiffness loss."
        )

    if defective_state.primary_overloaded:
        raise MechanicalGroundTruthError(
            "Benchmark design error: defective primary must remain below its "
            "allowable force before bypass loss."
        )

    (
        primary_force_after_bypass_loss,
        primary_utilization_after_bypass_loss,
        primary_overloaded_after_bypass_loss,
    ) = force_after_single_branch_loss(
        defective_primary,
        total_force_n=total_force_n,
    )

    if not primary_overloaded_after_bypass_loss:
        raise MechanicalGroundTruthError(
            "Benchmark design error: primary must overload after bypass loss."
        )

    failure_sequence = (
        FailureStep(
            step_index=1,
            failed_member_id="bypass_branch",
            force_at_failure_n=defective_state.bypass_force_n,
            utilization=defective_state.bypass_utilization,
            surviving_member_ids=(
                "primary_branch",
                "downstream_member",
            ),
        ),
        FailureStep(
            step_index=2,
            failed_member_id="primary_branch",
            force_at_failure_n=primary_force_after_bypass_loss,
            utilization=primary_utilization_after_bypass_loss,
            surviving_member_ids=(
                "downstream_member",
            ),
        ),
    )

    return MechanicalGroundTruth(
        case_id="mechanical_02P_parallel_spring_ground_truth",
        healthy_primary=healthy_primary,
        defective_primary=defective_primary,
        bypass=bypass,
        downstream=downstream,
        load_case=load_case,
        healthy_state=healthy_state,
        defective_state=defective_state,
        failure_sequence=failure_sequence,
        metadata={
            "domain": "mechanical",
            "model": "linear_axial_springs",
            "load_control": "force_controlled",
            "pre_stressed": True,
            "roif_used_to_generate_ground_truth": False,
            "biological_semantics": False,
            "units": "SI",
        },
    )


def expected_numeric_results() -> Mapping[str, float]:
    return MappingProxyType(
        {
            "healthy_primary_stiffness_n_per_m": 1000.0,
            "defective_primary_stiffness_n_per_m": 250.0,
            "bypass_stiffness_n_per_m": 500.0,
            "downstream_stiffness_n_per_m": 800.0,
            "total_force_n": 240.0,
            "healthy_elongation_m": 0.16,
            "healthy_primary_force_n": 160.0,
            "healthy_bypass_force_n": 80.0,
            "defective_elongation_m": 0.32,
            "defective_primary_force_n": 80.0,
            "defective_bypass_force_n": 160.0,
            "primary_force_after_bypass_loss_n": 240.0,
        }
    )


def assert_ground_truth_consistency(
    ground_truth: MechanicalGroundTruth | None = None,
) -> None:
    gt = ground_truth or build_physics_ground_truth()
    expected = expected_numeric_results()

    checks = (
        (
            gt.healthy_primary.axial_stiffness_n_per_m,
            expected["healthy_primary_stiffness_n_per_m"],
        ),
        (
            gt.defective_primary.axial_stiffness_n_per_m,
            expected["defective_primary_stiffness_n_per_m"],
        ),
        (
            gt.bypass.axial_stiffness_n_per_m,
            expected["bypass_stiffness_n_per_m"],
        ),
        (
            gt.downstream.axial_stiffness_n_per_m,
            expected["downstream_stiffness_n_per_m"],
        ),
        (
            gt.load_case.total_force_n,
            expected["total_force_n"],
        ),
        (
            gt.healthy_state.elongation_m,
            expected["healthy_elongation_m"],
        ),
        (
            gt.healthy_state.primary_force_n,
            expected["healthy_primary_force_n"],
        ),
        (
            gt.healthy_state.bypass_force_n,
            expected["healthy_bypass_force_n"],
        ),
        (
            gt.defective_state.elongation_m,
            expected["defective_elongation_m"],
        ),
        (
            gt.defective_state.primary_force_n,
            expected["defective_primary_force_n"],
        ),
        (
            gt.defective_state.bypass_force_n,
            expected["defective_bypass_force_n"],
        ),
    )

    for actual, target in checks:
        if not isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12):
            raise MechanicalGroundTruthError(
                f"ground-truth mismatch: {actual!r} != {target!r}"
            )

    if gt.healthy_state.any_overload:
        raise MechanicalGroundTruthError(
            "Healthy benchmark must contain no overload."
        )

    if not gt.defective_state.bypass_overloaded:
        raise MechanicalGroundTruthError(
            "Defective benchmark must overload bypass first."
        )

    if gt.failure_sequence[0].failed_member_id != "bypass_branch":
        raise MechanicalGroundTruthError(
            "Bypass must be first physical failure after stiffness defect."
        )

    if gt.failure_sequence[1].failed_member_id != "primary_branch":
        raise MechanicalGroundTruthError(
            "Primary must fail second after bypass loss."
        )


__all__ = [
    "AxialMember",
    "FailureStep",
    "LoadCase",
    "MechanicalGroundTruth",
    "MechanicalGroundTruthError",
    "ParallelForceState",
    "assert_ground_truth_consistency",
    "build_physics_ground_truth",
    "expected_numeric_results",
    "force_after_single_branch_loss",
    "parallel_force_state",
]
