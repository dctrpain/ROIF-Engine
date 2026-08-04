from __future__ import annotations

"""
Tests for immutable rheological memory.

These tests define the contract for:

- validation;
- derived rheological indices;
- immutable evolution;
- serialization;
- material-snapshot import;
- backward compatibility with pre-rheology snapshots.
"""

from dataclasses import FrozenInstanceError

import pytest

from roif.history.rheology_memory import (
    RheologicalMemory,
    RheologicalMemoryError,
    RheologyModel,
    RheologyPhase,
)


def make_memory(
    **overrides,
) -> RheologicalMemory:
    data = {
        "memory_id": "memory-001",
        "target_id": "ELEMENT_A",
        "observation_time": 12.0,
        "rheology_time": 10.0,
        "creep_strain": 0.08,
        "peak_creep_strain": 0.10,
        "elastic_strain": 0.04,
        "residual_strain": 0.03,
        "recoverable_strain": 0.05,
        "total_strain": 0.12,
        "instantaneous_stiffness": 10.0,
        "relaxed_stiffness": 5.0,
        "creep_time_constant": 2.0,
        "applied_stress": 1.0,
        "relaxation_fraction": 0.50,
        "recovery_fraction": 0.20,
        "retained_fraction": 0.80,
        "confidence": 0.90,
        "phase": RheologyPhase.CREEPING,
        "model": RheologyModel.STANDARD_LINEAR_SOLID,
        "source_snapshot_id": "snapshot-001",
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)
    return RheologicalMemory(**data)


def test_rheological_memory_creation() -> None:
    memory = make_memory()

    assert memory.memory_id == "memory-001"
    assert memory.target_id == "ELEMENT_A"
    assert memory.observation_time == pytest.approx(12.0)
    assert memory.rheology_time == pytest.approx(10.0)

    assert memory.creep_strain == pytest.approx(0.08)
    assert memory.peak_creep_strain == pytest.approx(0.10)
    assert memory.elastic_strain == pytest.approx(0.04)
    assert memory.residual_strain == pytest.approx(0.03)
    assert memory.recoverable_strain == pytest.approx(0.05)
    assert memory.total_strain == pytest.approx(0.12)

    assert memory.phase is RheologyPhase.CREEPING
    assert (
        memory.model
        is RheologyModel.STANDARD_LINEAR_SOLID
    )


def test_memory_accepts_string_enum_values() -> None:
    memory = make_memory(
        phase="recovering",
        model="maxwell",
    )

    assert memory.phase is RheologyPhase.RECOVERING
    assert memory.model is RheologyModel.MAXWELL


def test_memory_is_immutable() -> None:
    memory = make_memory()

    with pytest.raises(FrozenInstanceError):
        memory.creep_strain = 0.20


def test_metadata_is_read_only() -> None:
    memory = make_memory()

    with pytest.raises(TypeError):
        memory.metadata["new"] = "value"


def test_metadata_keys_are_sorted() -> None:
    memory = make_memory(
        metadata={
            "z": 1,
            "a": 2,
        }
    )

    assert tuple(memory.metadata) == (
        "a",
        "z",
    )


def test_has_creep() -> None:
    assert make_memory().has_creep is True

    empty = make_memory(
        creep_strain=0.0,
        peak_creep_strain=0.0,
        residual_strain=0.0,
        retained_fraction=0.0,
    )
    assert empty.has_creep is False


def test_has_residual_memory() -> None:
    assert make_memory().has_residual_memory is True

    empty = make_memory(
        residual_strain=0.0,
        retained_fraction=0.0,
    )
    assert empty.has_residual_memory is False


def test_is_recovering_from_phase() -> None:
    memory = make_memory(
        phase=RheologyPhase.RECOVERING,
        recoverable_strain=0.0,
        recovery_fraction=1.0,
    )

    assert memory.is_recovering is True


def test_is_recovering_from_state() -> None:
    memory = make_memory(
        phase=RheologyPhase.UNKNOWN,
        recoverable_strain=0.02,
        recovery_fraction=0.50,
    )

    assert memory.is_recovering is True


def test_is_relaxed() -> None:
    assert make_memory(
        phase=RheologyPhase.RELAXED,
    ).is_relaxed is True

    assert make_memory(
        phase=RheologyPhase.UNKNOWN,
        relaxation_fraction=1.0,
    ).is_relaxed is True


def test_stiffness_ratio() -> None:
    memory = make_memory(
        instantaneous_stiffness=10.0,
        relaxed_stiffness=4.0,
    )

    assert memory.stiffness_ratio == pytest.approx(0.4)
    assert memory.relaxable_fraction == pytest.approx(0.6)


def test_zero_stiffness_has_zero_ratios() -> None:
    memory = make_memory(
        instantaneous_stiffness=0.0,
        relaxed_stiffness=0.0,
    )

    assert memory.stiffness_ratio == pytest.approx(0.0)
    assert memory.relaxable_fraction == pytest.approx(0.0)


def test_strain_retention_ratio() -> None:
    memory = make_memory(
        creep_strain=0.06,
        residual_strain=0.04,
        peak_creep_strain=0.10,
    )

    assert memory.strain_retention_ratio == pytest.approx(0.60)


def test_residual_strain_can_dominate_retention_ratio() -> None:
    memory = make_memory(
        creep_strain=0.03,
        residual_strain=0.07,
        peak_creep_strain=0.10,
    )

    assert memory.strain_retention_ratio == pytest.approx(0.70)


def test_zero_peak_has_zero_retention_ratio() -> None:
    memory = make_memory(
        creep_strain=0.0,
        peak_creep_strain=0.0,
        residual_strain=0.0,
    )

    assert memory.strain_retention_ratio == pytest.approx(0.0)


def test_recovery_remaining() -> None:
    memory = make_memory(
        recovery_fraction=0.25,
    )

    assert memory.recovery_remaining == pytest.approx(0.75)


def test_normalized_age() -> None:
    memory = make_memory(
        rheology_time=6.0,
        creep_time_constant=2.0,
    )

    assert memory.normalized_age == pytest.approx(0.75)


def test_zero_time_constant_has_zero_normalized_age() -> None:
    memory = make_memory(
        creep_time_constant=0.0,
    )

    assert memory.normalized_age == pytest.approx(0.0)


def test_relaxation_index_is_bounded() -> None:
    memory = make_memory()

    assert 0.0 <= memory.relaxation_index <= 1.0


def test_memory_index_is_bounded() -> None:
    memory = make_memory()

    assert 0.0 <= memory.memory_index <= 1.0


def test_confidence_scales_memory_index() -> None:
    high = make_memory(
        confidence=1.0,
    )
    low = make_memory(
        confidence=0.25,
    )

    assert low.memory_index < high.memory_index


def test_signed_memory_strain_prefers_residual() -> None:
    memory = make_memory(
        creep_strain=0.08,
        residual_strain=0.03,
    )

    assert memory.signed_memory_strain == pytest.approx(0.03)


def test_signed_memory_strain_falls_back_to_creep() -> None:
    memory = make_memory(
        creep_strain=-0.08,
        residual_strain=0.0,
    )

    assert memory.signed_memory_strain == pytest.approx(-0.08)


def test_evolve_creates_new_revision() -> None:
    current = make_memory()

    evolved = current.evolve(
        memory_id="memory-002",
        observation_time=20.0,
        rheology_time=18.0,
        creep_strain=0.09,
        recovery_fraction=0.30,
        phase=RheologyPhase.RECOVERING,
    )

    assert evolved is not current
    assert evolved.memory_id == "memory-002"
    assert evolved.parent_memory_id == current.memory_id
    assert evolved.observation_time == pytest.approx(20.0)
    assert evolved.rheology_time == pytest.approx(18.0)
    assert evolved.creep_strain == pytest.approx(0.09)
    assert evolved.peak_creep_strain == pytest.approx(0.10)
    assert evolved.recovery_fraction == pytest.approx(0.30)
    assert evolved.phase is RheologyPhase.RECOVERING

    assert current.parent_memory_id is None
    assert current.creep_strain == pytest.approx(0.08)


def test_evolve_updates_peak_automatically() -> None:
    current = make_memory(
        peak_creep_strain=0.10,
    )

    evolved = current.evolve(
        observation_time=15.0,
        creep_strain=0.14,
    )

    assert evolved.peak_creep_strain == pytest.approx(0.14)


def test_evolve_merges_metadata() -> None:
    current = make_memory(
        metadata={
            "source": "initial",
            "keep": True,
        }
    )

    evolved = current.evolve(
        observation_time=15.0,
        metadata={
            "source": "follow_up",
            "new": 1,
        },
    )

    assert dict(evolved.metadata) == {
        "keep": True,
        "new": 1,
        "source": "follow_up",
    }


def test_to_dict_contains_raw_and_derived_values() -> None:
    memory = make_memory()
    data = memory.to_dict()

    assert data["version"] == RheologicalMemory.VERSION
    assert data["memory_id"] == "memory-001"
    assert data["phase"] == "creeping"
    assert data["model"] == "standard_linear_solid"
    assert data["creep_strain"] == pytest.approx(0.08)

    assert data["derived"]["has_creep"] is True
    assert "memory_index" in data["derived"]
    assert "relaxation_index" in data["derived"]


def test_to_dict_is_deterministic() -> None:
    memory = make_memory()

    assert memory.to_dict() == memory.to_dict()


def test_from_dict_round_trip() -> None:
    original = make_memory()
    restored = RheologicalMemory.from_dict(
        original.to_dict()
    )

    assert restored.to_dict() == original.to_dict()


def test_from_dict_ignores_derived_values() -> None:
    data = make_memory().to_dict()
    data["derived"]["memory_index"] = -100.0

    restored = RheologicalMemory.from_dict(data)

    assert restored.memory_index >= 0.0


def test_from_material_snapshot() -> None:
    snapshot = {
        "parameters": {
            "stiffness": 10.0,
            "relaxed_stiffness": 5.0,
            "creep_time_constant": 0.5,
            "rheology_enabled": True,
        },
        "state": {
            "creep_strain": 0.08,
            "rheology_time": 3.0,
            "strain": 0.12,
            "stress": 1.0,
        },
    }

    memory = RheologicalMemory.from_material_snapshot(
        snapshot,
        target_id="ELEMENT_A",
        observation_time=4.0,
        phase=RheologyPhase.CREEPING,
        source_snapshot_id="snapshot-001",
    )

    assert memory.target_id == "ELEMENT_A"
    assert memory.creep_strain == pytest.approx(0.08)
    assert memory.peak_creep_strain == pytest.approx(0.08)
    assert memory.rheology_time == pytest.approx(3.0)
    assert memory.total_strain == pytest.approx(0.12)
    assert memory.applied_stress == pytest.approx(1.0)
    assert memory.instantaneous_stiffness == pytest.approx(10.0)
    assert memory.relaxed_stiffness == pytest.approx(5.0)
    assert memory.creep_time_constant == pytest.approx(0.5)
    assert memory.relaxation_fraction == pytest.approx(0.5)
    assert memory.retained_fraction == pytest.approx(1.0)
    assert (
        memory.model
        is RheologyModel.STANDARD_LINEAR_SOLID
    )


def test_from_pre_rheology_snapshot_is_backward_compatible() -> None:
    snapshot = {
        "parameters": {
            "stiffness": 10.0,
        },
        "state": {
            "strain": 0.05,
        },
    }

    memory = RheologicalMemory.from_material_snapshot(
        snapshot,
        target_id="ELEMENT_A",
        observation_time=1.0,
    )

    assert memory.creep_strain == pytest.approx(0.0)
    assert memory.peak_creep_strain == pytest.approx(0.0)
    assert memory.rheology_time == pytest.approx(0.0)
    assert memory.instantaneous_stiffness == pytest.approx(10.0)
    assert memory.relaxed_stiffness == pytest.approx(10.0)
    assert memory.model is RheologyModel.ELASTIC


def test_from_material_snapshot_uses_explicit_residual_values() -> None:
    snapshot = {
        "parameters": {
            "stiffness": 10.0,
            "relaxed_stiffness": 5.0,
            "rheology_enabled": True,
        },
        "state": {
            "creep_strain": 0.04,
            "rheology_time": 2.0,
        },
    }

    memory = RheologicalMemory.from_material_snapshot(
        snapshot,
        target_id="ELEMENT_A",
        observation_time=3.0,
        residual_strain=0.02,
        recoverable_strain=0.02,
        recovery_fraction=0.50,
        retained_fraction=0.40,
    )

    assert memory.residual_strain == pytest.approx(0.02)
    assert memory.recoverable_strain == pytest.approx(0.02)
    assert memory.recovery_fraction == pytest.approx(0.50)
    assert memory.retained_fraction == pytest.approx(0.40)


@pytest.mark.parametrize(
    "field_name,bad_value",
    (
        ("observation_time", -1.0),
        ("rheology_time", -1.0),
        ("peak_creep_strain", -0.01),
        ("recoverable_strain", -0.01),
        ("instantaneous_stiffness", -1.0),
        ("relaxed_stiffness", -1.0),
        ("creep_time_constant", -1.0),
        ("relaxation_fraction", -0.01),
        ("relaxation_fraction", 1.01),
        ("recovery_fraction", -0.01),
        ("recovery_fraction", 1.01),
        ("retained_fraction", -0.01),
        ("retained_fraction", 1.01),
        ("confidence", -0.01),
        ("confidence", 1.01),
    ),
)
def test_memory_rejects_invalid_values(
    field_name: str,
    bad_value: float,
) -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match=field_name,
    ):
        make_memory(
            **{
                field_name: bad_value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "observation_time",
        "rheology_time",
        "creep_strain",
        "peak_creep_strain",
        "elastic_strain",
        "residual_strain",
        "recoverable_strain",
        "total_strain",
        "instantaneous_stiffness",
        "relaxed_stiffness",
        "creep_time_constant",
        "applied_stress",
        "relaxation_fraction",
        "recovery_fraction",
        "retained_fraction",
        "confidence",
    ),
)
def test_memory_rejects_nonfinite_values(
    field_name: str,
) -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match=field_name,
    ):
        make_memory(
            **{
                field_name: float("nan"),
            }
        )


def test_peak_must_cover_current_creep() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="peak_creep_strain",
    ):
        make_memory(
            creep_strain=0.20,
            peak_creep_strain=0.10,
        )


def test_relaxed_stiffness_cannot_exceed_instantaneous() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="relaxed_stiffness",
    ):
        make_memory(
            instantaneous_stiffness=5.0,
            relaxed_stiffness=6.0,
        )


def test_memory_rejects_self_parent() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="cannot be its own parent",
    ):
        make_memory(
            memory_id="same-id",
            parent_memory_id="same-id",
        )


def test_memory_rejects_empty_target_id() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="target_id",
    ):
        make_memory(
            target_id=" ",
        )


def test_memory_rejects_invalid_phase() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="unsupported rheology phase",
    ):
        make_memory(
            phase="unsupported",
        )


def test_memory_rejects_invalid_model() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="unsupported rheology model",
    ):
        make_memory(
            model="unsupported",
        )


def test_from_dict_rejects_nonmapping() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="must be a mapping",
    ):
        RheologicalMemory.from_dict("memory")


def test_from_material_snapshot_rejects_nonmapping() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="must be a mapping",
    ):
        RheologicalMemory.from_material_snapshot(
            "snapshot",
            target_id="ELEMENT_A",
            observation_time=0.0,
        )


def test_from_material_snapshot_rejects_invalid_sections() -> None:
    with pytest.raises(
        RheologicalMemoryError,
        match="snapshot parameters",
    ):
        RheologicalMemory.from_material_snapshot(
            {
                "parameters": "invalid",
                "state": {},
            },
            target_id="ELEMENT_A",
            observation_time=0.0,
        )

    with pytest.raises(
        RheologicalMemoryError,
        match="snapshot state",
    ):
        RheologicalMemory.from_material_snapshot(
            {
                "parameters": {},
                "state": "invalid",
            },
            target_id="ELEMENT_A",
            observation_time=0.0,
        )
