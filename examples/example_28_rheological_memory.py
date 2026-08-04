from __future__ import annotations

"""
Example 28 — Rheological memory becomes structural history.

Development level
-----------------
Research prototype / controlled computational demonstration.

This example demonstrates the complete ROIF chain:

    Material SLS state
        -> RheologicalMemory revisions
        -> IrreversibleChange
        -> HistoryPattern
        -> StructuralSignature

The example uses synthetic loading and recovery. It demonstrates software
behaviour and memory propagation; it is not clinical validation.
"""

import json
from pathlib import Path
from typing import Any

from core.material import Material, MaterialParameters
from roif.history.event import (
    HistoryTarget,
    HistoryTargetKind,
    ReversibilityClass,
    StateDelta,
    TimeScale,
)
from roif.history.history_pattern import HistoryPattern
from roif.history.irreversible_change import (
    FunctionalEffect,
    IrreversibleChange,
    IrreversibleChangeKind,
    TracePersistence,
)
from roif.history.rheology_memory import (
    RheologicalMemory,
    RheologyModel,
    RheologyPhase,
)
from roif.history.structural_signature import StructuralSignature


EXAMPLE_VERSION = "1.0.0"
TARGET_ID = "SLS_ELEMENT_28"

LOAD_EXTENSION = 0.12
LOAD_DURATION = 10.0
RECOVERY_DURATION = 6.0
DT = 0.05


def simulate_rheological_history() -> tuple[
    Material,
    float,
    float,
]:
    """
    Run a deterministic loading/recovery sequence.

    Returns
    -------
    material:
        Material in its final recovered state.
    peak_creep:
        Maximum creep strain reached during loading.
    loaded_creep:
        Creep strain at the end of sustained loading.
    """

    material = Material(
        name="example_28_sls_material",
        parameters=MaterialParameters(
            stiffness=10.0,
            damping=0.5,
            recovery_rate=0.0,
            fatigue_rate=0.0,
            remodeling_rate=0.0,
            production_rate=0.0,
            damage_rate=0.0,
            pretension_rate=0.0,
            energy_decay_rate=0.0,
            overload_threshold=100.0,
            failure_threshold=1000.0,
            reference_force=1.0,
            reference_strain=1.0,
            rheology_enabled=True,
            relaxed_stiffness=5.0,
            creep_time_constant=2.0,
        ),
        reference_length=1.0,
    )

    peak_creep = 0.0

    load_steps = round(LOAD_DURATION / DT)
    for _ in range(load_steps):
        material.update_rheology(
            dt=DT,
            extension=LOAD_EXTENSION,
        )
        peak_creep = max(
            peak_creep,
            abs(material.creep_strain),
        )

    loaded_creep = material.creep_strain

    recovery_steps = round(RECOVERY_DURATION / DT)
    for _ in range(recovery_steps):
        material.update_rheology(
            dt=DT,
            extension=0.0,
        )

    return material, peak_creep, loaded_creep


def build_memory_revisions(
    *,
    material: Material,
    peak_creep: float,
    loaded_creep: float,
) -> tuple[RheologicalMemory, RheologicalMemory]:
    """Create immutable loaded and recovered memory revisions."""

    parameters = material.parameters

    loaded_memory = RheologicalMemory(
        memory_id="example-28-memory-loaded",
        target_id=TARGET_ID,
        observation_time=LOAD_DURATION,
        rheology_time=LOAD_DURATION,
        creep_strain=loaded_creep,
        peak_creep_strain=peak_creep,
        elastic_strain=LOAD_EXTENSION - loaded_creep,
        residual_strain=0.0,
        recoverable_strain=loaded_creep,
        total_strain=LOAD_EXTENSION,
        instantaneous_stiffness=parameters.stiffness,
        relaxed_stiffness=float(
            parameters.relaxed_stiffness
            if parameters.relaxed_stiffness is not None
            else parameters.stiffness
        ),
        creep_time_constant=parameters.creep_time_constant,
        applied_stress=1.0,
        relaxation_fraction=(
            1.0
            - float(parameters.relaxed_stiffness)
            / parameters.stiffness
        ),
        recovery_fraction=0.0,
        retained_fraction=1.0,
        confidence=1.0,
        phase=RheologyPhase.CREEPING,
        model=RheologyModel.STANDARD_LINEAR_SOLID,
        source_snapshot_id="example-28-loaded-snapshot",
        metadata={
            "stage": "sustained_load",
            "synthetic": True,
        },
    )

    recovered_creep = material.creep_strain
    recovered_amount = max(
        loaded_creep - recovered_creep,
        0.0,
    )
    recovery_fraction = (
        recovered_amount / loaded_creep
        if loaded_creep > 0.0
        else 0.0
    )
    retained_fraction = (
        recovered_creep / peak_creep
        if peak_creep > 0.0
        else 0.0
    )

    recovered_memory = loaded_memory.evolve(
        memory_id="example-28-memory-recovered",
        observation_time=LOAD_DURATION + RECOVERY_DURATION,
        rheology_time=LOAD_DURATION + RECOVERY_DURATION,
        creep_strain=recovered_creep,
        peak_creep_strain=peak_creep,
        elastic_strain=0.0,
        residual_strain=recovered_creep,
        recoverable_strain=0.0,
        total_strain=recovered_creep,
        applied_stress=0.0,
        recovery_fraction=min(max(recovery_fraction, 0.0), 1.0),
        retained_fraction=min(max(retained_fraction, 0.0), 1.0),
        phase=RheologyPhase.RESIDUAL,
        metadata={
            "stage": "post_unloading",
            "synthetic": True,
        },
    )

    return loaded_memory, recovered_memory


def build_irreversible_change(
    memory: RheologicalMemory,
) -> IrreversibleChange:
    """Convert retained rheological memory into structural history."""

    target = HistoryTarget(
        kind=HistoryTargetKind.ELEMENT,
        target_id=TARGET_ID,
        label="Example 28 SLS element",
        metadata={
            "domain": "synthetic_material",
        },
    )

    return IrreversibleChange(
        change_id="example-28-creep-change",
        kind=IrreversibleChangeKind.CREEP,
        source_event_id="example-28-load-recovery-event",
        target=target,
        delta=StateDelta(
            quantity="creep_strain",
            before=0.0,
            after=memory.residual_strain,
            units="strain",
            channel="rheology",
            metadata={
                "peak_creep_strain": memory.peak_creep_strain,
            },
        ),
        onset_time=0.0,
        recorded_time=memory.observation_time,
        retained_fraction=max(
            memory.retained_fraction,
            1e-9,
        ),
        permanence=TracePersistence.LONG_LIVED,
        characteristic_time=max(
            memory.creep_time_constant,
            1e-9,
        ),
        time_scale=TimeScale.MEDIUM,
        memory_strength=memory.memory_index,
        capacity_effect=-min(
            abs(memory.residual_strain),
            1.0,
        ),
        functional_effect=FunctionalEffect.HARMFUL,
        reversibility=ReversibilityClass.PARTIALLY_REVERSIBLE,
        plane_ids=(
            "mechanical",
            "rheological",
        ),
        agent_ids=(
            "sustained_external_load",
        ),
        description=(
            "Retained creep trace after sustained loading "
            "and partial recovery."
        ),
        metadata={
            "example": 28,
            "synthetic": True,
        },
        rheology_memory=memory,
    )


def build_signatures(
    change: IrreversibleChange,
) -> tuple[StructuralSignature, StructuralSignature]:
    """Build rheological and no-memory reference signatures."""

    rheological_pattern = HistoryPattern(
        pattern_id="example-28-rheological-pattern",
        changes=(change,),
        label="SLS creep and recovery history",
        description=(
            "Synthetic sustained load followed by partial recovery."
        ),
        external_cause_ids=("sustained_external_load",),
        metadata={
            "example": 28,
            "development_level": "research_prototype",
        },
    )

    rheological_signature = StructuralSignature.from_pattern(
        rheological_pattern,
        label="Example 28 rheological signature",
    )

    reference_change = IrreversibleChange(
        change_id="example-28-reference-change",
        kind=IrreversibleChangeKind.CREEP,
        source_event_id="example-28-reference-event",
        target=change.target,
        delta=change.delta,
        onset_time=change.onset_time,
        recorded_time=change.recorded_time,
        retained_fraction=change.retained_fraction,
        permanence=change.permanence,
        characteristic_time=change.characteristic_time,
        time_scale=change.time_scale,
        memory_strength=change.memory_strength,
        capacity_effect=change.capacity_effect,
        functional_effect=change.functional_effect,
        reversibility=change.reversibility,
        plane_ids=change.plane_ids,
        agent_ids=change.agent_ids,
        description="Reference trace without attached rheological memory.",
        metadata={
            "example": 28,
            "reference_without_rheology": True,
        },
        rheology_memory=None,
    )

    reference_pattern = HistoryPattern(
        pattern_id="example-28-reference-pattern",
        changes=(reference_change,),
        label="Reference history without rheological memory",
    )

    reference_signature = StructuralSignature.from_pattern(
        reference_pattern,
        label="Example 28 reference signature",
    )

    return rheological_signature, reference_signature


def build_report() -> dict[str, Any]:
    material, peak_creep, loaded_creep = (
        simulate_rheological_history()
    )
    loaded_memory, recovered_memory = build_memory_revisions(
        material=material,
        peak_creep=peak_creep,
        loaded_creep=loaded_creep,
    )
    change = build_irreversible_change(
        recovered_memory
    )
    signature, reference_signature = build_signatures(
        change
    )
    comparison = reference_signature.compare(signature)

    dominant_rheology = signature.dominant_rheology
    dominant_name = (
        dominant_rheology[0]
        if dominant_rheology is not None
        else None
    )
    dominant_weight = (
        dominant_rheology[1]
        if dominant_rheology is not None
        else 0.0
    )

    return {
        "example": 28,
        "version": EXAMPLE_VERSION,
        "project_stage": "research_prototype",
        "validation_scope": "controlled_synthetic_computational",
        "clinical_validation": False,
        "simulation": {
            "load_extension": LOAD_EXTENSION,
            "load_duration": LOAD_DURATION,
            "recovery_duration": RECOVERY_DURATION,
            "dt": DT,
            "loaded_creep_strain": loaded_creep,
            "peak_creep_strain": peak_creep,
            "final_creep_strain": material.creep_strain,
        },
        "memory_revisions": {
            "loaded": loaded_memory.to_dict(),
            "recovered": recovered_memory.to_dict(),
            "parent_link_preserved": (
                recovered_memory.parent_memory_id
                == loaded_memory.memory_id
            ),
        },
        "irreversible_change": change.to_dict(),
        "structural_signature": signature.to_dict(),
        "comparison_with_no_memory_reference": (
            comparison.to_dict()
        ),
        "summary": {
            "material_memory_recorded": (
                recovered_memory.has_creep
            ),
            "residual_trace_recorded": (
                recovered_memory.has_residual_memory
            ),
            "irreversible_change_created": (
                change.has_rheological_memory
            ),
            "structural_signature_updated": (
                signature.has_rheological_memory
            ),
            "previous_revision_preserved": (
                recovered_memory.parent_memory_id
                == loaded_memory.memory_id
            ),
            "dominant_rheology": dominant_name,
            "dominant_rheology_weight": dominant_weight,
            "rheological_memory_index": (
                recovered_memory.memory_index
            ),
            "signature_distance_from_no_memory": (
                comparison.distance
            ),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    simulation = report["simulation"]
    summary = report["summary"]

    print("=" * 72)
    print("ROIF EXAMPLE 28 — RHEOLOGICAL MEMORY")
    print("=" * 72)
    print("Development level : Research prototype")
    print("Evidence type     : Controlled synthetic computation")
    print("Clinical evidence : No")
    print("-" * 72)
    print(
        "Loaded creep strain       : "
        f"{simulation['loaded_creep_strain']:.6f}"
    )
    print(
        "Peak creep strain         : "
        f"{simulation['peak_creep_strain']:.6f}"
    )
    print(
        "Residual creep strain     : "
        f"{simulation['final_creep_strain']:.6f}"
    )
    print(
        "Rheological memory index  : "
        f"{summary['rheological_memory_index']:.6f}"
    )
    print(
        "Dominant rheology         : "
        f"{summary['dominant_rheology']}"
    )
    print(
        "Signature distance        : "
        f"{summary['signature_distance_from_no_memory']:.6f}"
    )
    print("-" * 72)
    print(
        "Material memory recorded  : "
        f"{summary['material_memory_recorded']}"
    )
    print(
        "Residual trace recorded   : "
        f"{summary['residual_trace_recorded']}"
    )
    print(
        "IrreversibleChange linked : "
        f"{summary['irreversible_change_created']}"
    )
    print(
        "StructuralSignature linked: "
        f"{summary['structural_signature_updated']}"
    )
    print(
        "Previous revision preserved: "
        f"{summary['previous_revision_preserved']}"
    )
    print("=" * 72)


def main() -> int:
    report = build_report()
    print_report(report)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / "example_28_rheological_memory.json"
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved report: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
