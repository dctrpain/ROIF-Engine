"""
Tests for roif.roif_materials.

The suite fixes the universal ROIF material-layer contract:

- biological and non-biological material families;
- immutable descriptors, parameters, states, laws, and traces;
- elastic, nonlinear, viscous, rheological, plastic, active, thermal,
  moisture, damage, and fatigue responses;
- hybrid constitutive models;
- capacity degradation and failure detection;
- material-state advancement;
- multiplicative plane factors;
- conversion of core.material-style snapshots;
- immutable material lookup tables.

These tests validate a reduced-order research representation. They do not
claim that one scalar constitutive law replaces a validated finite-element
material model.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
import math

import pytest

from roif.roif_entities import Vector3
from roif.roif_materials import (
    AnisotropyKind,
    ConstitutiveLaw,
    ConstitutiveLawKind,
    FailureMode,
    LawContribution,
    MaterialDescriptor,
    MaterialEffect,
    MaterialFamily,
    MaterialInput,
    MaterialModel,
    MaterialParameters,
    MaterialPhase,
    MaterialResponse,
    MaterialState,
    MaterialTrace,
    ROIFMaterialError,
    ResponseMode,
    advance_material,
    default_material_model,
    detect_failure_modes,
    evaluate_law,
    evaluate_material,
    material_models_by_id,
    material_plane_factors,
    material_state_from_snapshot,
    model_from_core_material,
    response_to_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_descriptor(
    *,
    material_id: str = "steel",
    family: MaterialFamily = MaterialFamily.METAL,
    anisotropy: AnisotropyKind = AnisotropyKind.ISOTROPIC,
    active_material: bool = False,
    biological: bool = False,
) -> MaterialDescriptor:
    return MaterialDescriptor(
        material_id=material_id,
        name=material_id.replace("_", " ").title(),
        family=family,
        anisotropy=anisotropy,
        active_material=active_material,
        biological=biological,
        metadata={"source": "test"},
    )


def make_parameters(**overrides: float | None) -> MaterialParameters:
    values: dict[str, object] = {
        "density": 1000.0,
        "elastic_modulus": 100.0,
        "shear_modulus": 40.0,
        "bulk_modulus": 80.0,
        "damping": 2.0,
        "reference_stress": 100.0,
        "reference_strain": 0.1,
    }
    values.update(overrides)
    return MaterialParameters(**values)


def make_model(
    *,
    family: MaterialFamily = MaterialFamily.METAL,
    parameters: MaterialParameters | None = None,
    laws: tuple[ConstitutiveLaw, ...] | None = None,
    state: MaterialState | None = None,
    active_material: bool = False,
) -> MaterialModel:
    return MaterialModel(
        descriptor=make_descriptor(
            family=family,
            active_material=active_material,
            biological=family
            in {
                MaterialFamily.BIOLOGICAL_TISSUE,
                MaterialFamily.MUSCLE,
                MaterialFamily.TENDON,
                MaterialFamily.LIGAMENT,
                MaterialFamily.FASCIA,
                MaterialFamily.BONE,
                MaterialFamily.CARTILAGE,
                MaterialFamily.NERVE,
                MaterialFamily.VESSEL,
                MaterialFamily.ORGAN_TISSUE,
            },
        ),
        parameters=parameters or make_parameters(),
        laws=laws
        or (
            ConstitutiveLaw(
                law_id="elastic",
                kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            ),
        ),
        state=state or MaterialState(),
    )


class FakeCoreMaterial:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> dict[str, object]:
        return dict(self._snapshot)


# ---------------------------------------------------------------------------
# Enumerations and descriptor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family",
    [
        MaterialFamily.METAL,
        MaterialFamily.CONCRETE,
        MaterialFamily.TIMBER,
        MaterialFamily.POLYMER,
        MaterialFamily.COMPOSITE,
        MaterialFamily.MUSCLE,
        MaterialFamily.TENDON,
        MaterialFamily.FLUID,
        MaterialFamily.SOFTWARE,
        MaterialFamily.INFORMATION,
    ],
)
def test_material_families_are_supported(
    family: MaterialFamily,
) -> None:
    descriptor = make_descriptor(
        material_id=family.value,
        family=family,
    )

    assert descriptor.family is family


def test_descriptor_creation_and_metadata_freezing() -> None:
    descriptor = make_descriptor()

    assert descriptor.material_id == "steel"
    assert descriptor.family is MaterialFamily.METAL
    assert isinstance(descriptor.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        descriptor.metadata["source"] = "changed"


def test_descriptor_is_immutable() -> None:
    descriptor = make_descriptor()

    with pytest.raises(FrozenInstanceError):
        descriptor.name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("material_id", ""),
        ("name", " "),
    ],
)
def test_descriptor_rejects_empty_text(
    field_name: str,
    value: str,
) -> None:
    kwargs = {
        "material_id": "material",
        "name": "Material",
        "family": MaterialFamily.GENERIC,
    }
    kwargs[field_name] = value

    with pytest.raises(ROIFMaterialError):
        MaterialDescriptor(**kwargs)


# ---------------------------------------------------------------------------
# MaterialParameters
# ---------------------------------------------------------------------------


def test_default_parameters_are_valid() -> None:
    parameters = MaterialParameters()

    assert parameters.elastic_modulus == pytest.approx(1.0)
    assert math.isinf(parameters.yield_stress)
    assert math.isinf(parameters.tensile_strength)
    assert math.isinf(parameters.failure_strain)


def test_parameters_accept_finite_failure_thresholds() -> None:
    parameters = make_parameters(
        yield_stress=50.0,
        tensile_strength=80.0,
        compressive_strength=120.0,
        shear_strength=40.0,
        failure_strain=0.4,
    )

    assert parameters.yield_stress == pytest.approx(50.0)
    assert parameters.failure_strain == pytest.approx(0.4)


def test_extra_parameters_are_read_only_and_numeric() -> None:
    parameters = MaterialParameters(
        extra={"hardening_modulus": 5.0},
    )

    assert isinstance(parameters.extra, MappingProxyType)
    assert parameters.extra["hardening_modulus"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("density", 0.0),
        ("elastic_modulus", 0.0),
        ("shear_modulus", -1.0),
        ("bulk_modulus", 0.0),
        ("creep_time_constant", 0.0),
        ("reference_stress", 0.0),
        ("damping", -0.1),
        ("yield_stress", -1.0),
        ("failure_strain", -1.0),
    ],
)
def test_parameters_reject_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialParameters(**{field_name: value})


@pytest.mark.parametrize("value", [-1.0, 0.5, 1.0])
def test_parameters_reject_invalid_poisson_ratio(
    value: float,
) -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialParameters(poisson_ratio=value)


def test_relaxed_modulus_cannot_exceed_instantaneous_modulus() -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialParameters(
            elastic_modulus=10.0,
            relaxed_modulus=11.0,
        )


# ---------------------------------------------------------------------------
# MaterialState
# ---------------------------------------------------------------------------


def test_default_material_state() -> None:
    state = MaterialState()

    assert state.phase is MaterialPhase.INSTANTANEOUS
    assert state.integrity == pytest.approx(1.0)
    assert state.available_capacity == pytest.approx(1.0)
    assert state.failed is False


def test_available_capacity_is_multiplicative() -> None:
    state = MaterialState(
        integrity=0.9,
        damage=0.2,
        fatigue=0.25,
        corrosion=0.1,
        aging=0.2,
    )

    expected = 0.9 * 0.8 * 0.75 * 0.9 * 0.8

    assert state.available_capacity == pytest.approx(expected)


def test_retained_fraction_uses_residual_strain() -> None:
    state = MaterialState(
        strain=0.2,
        residual_strain=0.05,
    )

    assert state.retained_fraction == pytest.approx(0.25)


def test_zero_total_strain_has_zero_retained_fraction() -> None:
    state = MaterialState(
        strain=0.0,
        residual_strain=0.1,
    )

    assert state.retained_fraction == pytest.approx(0.0)


def test_directional_state_is_read_only() -> None:
    state = MaterialState(
        directional_state={"longitudinal": 0.8},
    )

    assert isinstance(state.directional_state, MappingProxyType)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("time", -1.0),
        ("moisture", -0.1),
        ("integrity", 1.1),
        ("damage", -0.1),
        ("fatigue", 1.1),
        ("activation", -0.1),
        ("energy", -1.0),
    ],
)
def test_state_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialState(**{field_name: value})


def test_failed_state_requires_failure_mode() -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialState(
            failed=True,
            failure_mode=FailureMode.NONE,
        )


def test_failed_state_with_mode_is_valid() -> None:
    state = MaterialState(
        failed=True,
        failure_mode=FailureMode.TENSILE_RUPTURE,
        phase=MaterialPhase.FAILED,
    )

    assert state.failed is True


# ---------------------------------------------------------------------------
# ConstitutiveLaw and MaterialModel
# ---------------------------------------------------------------------------


def test_constitutive_law_creation() -> None:
    law = ConstitutiveLaw(
        law_id="elastic",
        kind=ConstitutiveLawKind.LINEAR_ELASTIC,
        response_modes=(
            ResponseMode.TENSION,
            ResponseMode.COMPRESSION,
        ),
        parameters={"elastic_modulus": 200.0},
    )

    assert law.parameters["elastic_modulus"] == pytest.approx(200.0)
    assert isinstance(law.parameters, MappingProxyType)


def test_law_requires_response_mode() -> None:
    with pytest.raises(ROIFMaterialError):
        ConstitutiveLaw(
            law_id="law",
            kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            response_modes=(),
        )


def test_material_model_creation() -> None:
    model = make_model()

    assert model.descriptor.family is MaterialFamily.METAL
    assert model.law("elastic").kind is ConstitutiveLawKind.LINEAR_ELASTIC


def test_model_normalizes_laws_to_tuple() -> None:
    model = MaterialModel(
        descriptor=make_descriptor(),
        laws=[
            ConstitutiveLaw(
                law_id="elastic",
                kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            )
        ],  # type: ignore[arg-type]
    )

    assert isinstance(model.laws, tuple)


def test_model_rejects_duplicate_law_ids() -> None:
    law = ConstitutiveLaw(
        law_id="same",
        kind=ConstitutiveLawKind.LINEAR_ELASTIC,
    )

    with pytest.raises(ROIFMaterialError):
        MaterialModel(
            descriptor=make_descriptor(),
            laws=(law, law),
        )


def test_model_unknown_law_raises_key_error() -> None:
    with pytest.raises(KeyError):
        make_model().law("unknown")


# ---------------------------------------------------------------------------
# MaterialInput
# ---------------------------------------------------------------------------


def test_material_input_creation() -> None:
    material_input = MaterialInput(
        strain=0.1,
        strain_rate=0.2,
        delta_time=0.5,
        temperature=30.0,
        moisture=0.3,
        activation=0.8,
        direction=Vector3(0.0, 1.0, 0.0),
    )

    assert material_input.activation == pytest.approx(0.8)
    assert material_input.direction == Vector3(0.0, 1.0, 0.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("delta_time", 0.0),
        ("delta_time", -1.0),
        ("activation", -0.1),
        ("activation", 1.1),
    ],
)
def test_material_input_rejects_invalid_values(
    field_name: str,
    value: float,
) -> None:
    kwargs: dict[str, object] = {"strain": 0.1}
    kwargs[field_name] = value

    with pytest.raises(ROIFMaterialError):
        MaterialInput(**kwargs)


# ---------------------------------------------------------------------------
# Individual constitutive laws
# ---------------------------------------------------------------------------


def test_linear_elastic_law() -> None:
    law = ConstitutiveLaw(
        law_id="elastic",
        kind=ConstitutiveLawKind.LINEAR_ELASTIC,
    )
    contribution = evaluate_law(
        law,
        make_parameters(elastic_modulus=100.0),
        MaterialState(),
        MaterialInput(strain=0.02),
    )

    assert contribution.stress == pytest.approx(2.0)
    assert contribution.tangent_modulus == pytest.approx(100.0)


def test_law_weight_scales_response() -> None:
    law = ConstitutiveLaw(
        law_id="elastic",
        kind=ConstitutiveLawKind.LINEAR_ELASTIC,
        weight=0.5,
    )
    contribution = evaluate_law(
        law,
        make_parameters(elastic_modulus=100.0),
        MaterialState(),
        MaterialInput(strain=0.02),
    )

    assert contribution.stress == pytest.approx(1.0)
    assert contribution.tangent_modulus == pytest.approx(50.0)


def test_inactive_law_has_zero_contribution() -> None:
    law = ConstitutiveLaw(
        law_id="inactive",
        kind=ConstitutiveLawKind.LINEAR_ELASTIC,
        active=False,
    )

    contribution = evaluate_law(
        law,
        make_parameters(),
        MaterialState(),
        MaterialInput(strain=1.0),
    )

    assert contribution.stress == pytest.approx(0.0)


def test_response_mode_filters_law() -> None:
    law = ConstitutiveLaw(
        law_id="active",
        kind=ConstitutiveLawKind.ACTIVE_CONTRACTILE,
        response_modes=(ResponseMode.ACTIVE,),
    )

    contribution = evaluate_law(
        law,
        make_parameters(active_stress_scale=10.0),
        MaterialState(activation=1.0),
        MaterialInput(
            strain=0.0,
            response_mode=ResponseMode.TENSION,
        ),
    )

    assert contribution.stress == pytest.approx(0.0)


def test_nonlinear_elastic_law() -> None:
    law = ConstitutiveLaw(
        law_id="nonlinear",
        kind=ConstitutiveLawKind.NONLINEAR_ELASTIC,
        parameters={"exponent": 2.0},
    )

    contribution = evaluate_law(
        law,
        make_parameters(elastic_modulus=100.0),
        MaterialState(),
        MaterialInput(strain=0.2),
    )

    assert contribution.stress == pytest.approx(4.0)
    assert contribution.tangent_modulus == pytest.approx(40.0)


def test_viscoelastic_law_includes_rate_and_dissipation() -> None:
    law = ConstitutiveLaw(
        law_id="visco",
        kind=ConstitutiveLawKind.VISCOELASTIC,
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            elastic_modulus=100.0,
            viscosity=5.0,
        ),
        MaterialState(),
        MaterialInput(
            strain=0.1,
            strain_rate=2.0,
            delta_time=0.5,
        ),
    )

    assert contribution.stress == pytest.approx(20.0)
    assert contribution.dissipated_energy == pytest.approx(10.0)


def test_standard_linear_solid_records_creep_increment() -> None:
    law = ConstitutiveLaw(
        law_id="sls",
        kind=ConstitutiveLawKind.STANDARD_LINEAR_SOLID,
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            elastic_modulus=100.0,
            relaxed_modulus=50.0,
            creep_time_constant=2.0,
        ),
        MaterialState(creep_strain=0.0),
        MaterialInput(
            strain=0.1,
            delta_time=1.0,
        ),
    )

    assert contribution.stress > 0.0
    assert contribution.state_delta["creep_strain"] > 0.0


def test_elastoplastic_below_yield_is_elastic() -> None:
    law = ConstitutiveLaw(
        law_id="plastic",
        kind=ConstitutiveLawKind.ELASTOPLASTIC,
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            elastic_modulus=100.0,
            yield_stress=10.0,
        ),
        MaterialState(),
        MaterialInput(strain=0.05),
    )

    assert contribution.stress == pytest.approx(5.0)
    assert "plastic_strain" not in contribution.state_delta


def test_elastoplastic_above_yield_records_plasticity() -> None:
    law = ConstitutiveLaw(
        law_id="plastic",
        kind=ConstitutiveLawKind.ELASTOPLASTIC,
        parameters={"hardening_modulus": 10.0},
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            elastic_modulus=100.0,
            yield_stress=10.0,
        ),
        MaterialState(),
        MaterialInput(strain=0.2),
    )

    assert contribution.stress >= 10.0
    assert contribution.state_delta["plastic_strain"] > 0.0


def test_active_contractile_law() -> None:
    law = ConstitutiveLaw(
        law_id="active",
        kind=ConstitutiveLawKind.ACTIVE_CONTRACTILE,
        response_modes=(ResponseMode.ACTIVE,),
    )

    contribution = evaluate_law(
        law,
        make_parameters(active_stress_scale=50.0),
        MaterialState(),
        MaterialInput(
            strain=0.0,
            strain_rate=0.0,
            activation=0.8,
            response_mode=ResponseMode.ACTIVE,
        ),
    )

    assert contribution.stress == pytest.approx(40.0)


def test_active_contractile_output_reduces_away_from_optimum() -> None:
    law = ConstitutiveLaw(
        law_id="active",
        kind=ConstitutiveLawKind.ACTIVE_CONTRACTILE,
        response_modes=(ResponseMode.ACTIVE,),
        parameters={
            "optimal_strain": 0.0,
            "length_width": 0.5,
        },
    )
    parameters = make_parameters(active_stress_scale=50.0)

    optimal = evaluate_law(
        law,
        parameters,
        MaterialState(),
        MaterialInput(
            strain=0.0,
            activation=1.0,
            response_mode=ResponseMode.ACTIVE,
        ),
    )
    nonoptimal = evaluate_law(
        law,
        parameters,
        MaterialState(),
        MaterialInput(
            strain=1.0,
            activation=1.0,
            response_mode=ResponseMode.ACTIVE,
        ),
    )

    assert nonoptimal.stress < optimal.stress


def test_thermal_expansion_changes_effective_stress() -> None:
    law = ConstitutiveLaw(
        law_id="thermal",
        kind=ConstitutiveLawKind.THERMAL_EXPANSION,
    )
    parameters = make_parameters(
        elastic_modulus=100.0,
        thermal_expansion=0.01,
        reference_temperature=20.0,
    )

    contribution = evaluate_law(
        law,
        parameters,
        MaterialState(),
        MaterialInput(
            strain=0.1,
            temperature=30.0,
        ),
    )

    assert contribution.stress == pytest.approx(0.0)


def test_moisture_swelling_changes_effective_stress() -> None:
    law = ConstitutiveLaw(
        law_id="moisture",
        kind=ConstitutiveLawKind.MOISTURE_SWELLING,
    )
    parameters = make_parameters(
        elastic_modulus=100.0,
        moisture_expansion=0.2,
        reference_moisture=0.0,
    )

    contribution = evaluate_law(
        law,
        parameters,
        MaterialState(),
        MaterialInput(
            strain=0.1,
            moisture=0.5,
        ),
    )

    assert contribution.stress == pytest.approx(0.0)


def test_damage_law_records_damage_increment() -> None:
    law = ConstitutiveLaw(
        law_id="damage",
        kind=ConstitutiveLawKind.DAMAGE,
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            damage_rate=0.5,
            reference_strain=0.1,
        ),
        MaterialState(),
        MaterialInput(
            strain=0.2,
            delta_time=1.0,
        ),
    )

    assert contribution.state_delta["damage"] == pytest.approx(0.5)


def test_fatigue_law_records_fatigue_increment() -> None:
    law = ConstitutiveLaw(
        law_id="fatigue",
        kind=ConstitutiveLawKind.FATIGUE,
    )

    contribution = evaluate_law(
        law,
        make_parameters(
            fatigue_rate=0.2,
            reference_strain=0.1,
        ),
        MaterialState(),
        MaterialInput(
            strain=0.2,
            delta_time=1.0,
        ),
    )

    assert contribution.state_delta["fatigue"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Complete material evaluation
# ---------------------------------------------------------------------------


def test_evaluate_material_linear_model() -> None:
    model = make_model(
        parameters=make_parameters(
            elastic_modulus=100.0,
            reference_stress=100.0,
        )
    )

    response = evaluate_material(
        model,
        MaterialInput(strain=0.1),
    )

    assert response.stress == pytest.approx(10.0)
    assert response.tangent_modulus == pytest.approx(100.0)
    assert response.force_factor == pytest.approx(0.1)
    assert response.state_after.time == pytest.approx(1.0)


def test_existing_damage_reduces_material_response() -> None:
    undamaged = make_model()
    damaged = make_model(
        state=MaterialState(
            integrity=1.0,
            damage=0.5,
        )
    )

    full_response = evaluate_material(
        undamaged,
        MaterialInput(strain=0.1),
    )
    reduced_response = evaluate_material(
        damaged,
        MaterialInput(strain=0.1),
    )

    assert reduced_response.stress < full_response.stress


def test_hybrid_laws_are_summed() -> None:
    laws = (
        ConstitutiveLaw(
            law_id="elastic_a",
            kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            weight=0.5,
        ),
        ConstitutiveLaw(
            law_id="elastic_b",
            kind=ConstitutiveLawKind.LINEAR_ELASTIC,
            weight=0.5,
        ),
    )
    model = make_model(
        parameters=make_parameters(elastic_modulus=100.0),
        laws=laws,
    )

    response = evaluate_material(
        model,
        MaterialInput(strain=0.1),
    )

    assert response.stress == pytest.approx(10.0)


def test_evaluation_updates_energy_and_history() -> None:
    response = evaluate_material(
        make_model(),
        MaterialInput(strain=0.1),
    )

    assert response.state_after.history == pytest.approx(0.1)
    assert response.state_after.energy > 0.0


def test_advance_material_returns_new_model() -> None:
    model = make_model()
    advanced, response = advance_material(
        model,
        MaterialInput(strain=0.1),
    )

    assert advanced is not model
    assert advanced.state == response.state_after
    assert model.state.time == pytest.approx(0.0)
    assert advanced.state.time == pytest.approx(1.0)


def test_response_and_contributions_are_immutable() -> None:
    response = evaluate_material(
        make_model(),
        MaterialInput(strain=0.1),
    )

    assert isinstance(response, MaterialResponse)
    assert isinstance(response.contributions, tuple)
    assert isinstance(response.metadata, MappingProxyType)

    with pytest.raises(FrozenInstanceError):
        response.stress = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


def test_tensile_failure_detection() -> None:
    model = make_model(
        parameters=make_parameters(
            elastic_modulus=100.0,
            tensile_strength=5.0,
        )
    )

    response = evaluate_material(
        model,
        MaterialInput(strain=0.1),
    )

    assert FailureMode.TENSILE_RUPTURE in response.failure_modes
    assert response.state_after.failed is True
    assert response.state_after.phase is MaterialPhase.FAILED


def test_compressive_failure_detection() -> None:
    model = make_model(
        parameters=make_parameters(
            elastic_modulus=100.0,
            compressive_strength=5.0,
        )
    )

    response = evaluate_material(
        model,
        MaterialInput(
            strain=-0.1,
            response_mode=ResponseMode.COMPRESSION,
        ),
    )

    assert FailureMode.COMPRESSIVE_CRUSHING in response.failure_modes


def test_shear_failure_requires_shear_mode() -> None:
    model = make_model(
        parameters=make_parameters(
            elastic_modulus=100.0,
            shear_strength=5.0,
        ),
        laws=(
            ConstitutiveLaw(
                law_id="shear",
                kind=ConstitutiveLawKind.LINEAR_ELASTIC,
                response_modes=(ResponseMode.SHEAR,),
            ),
        ),
    )

    response = evaluate_material(
        model,
        MaterialInput(
            strain=0.1,
            response_mode=ResponseMode.SHEAR,
        ),
    )

    assert FailureMode.SHEAR_FAILURE in response.failure_modes


def test_failure_strain_detection() -> None:
    model = make_model(
        parameters=make_parameters(
            failure_strain=0.05,
        )
    )

    response = evaluate_material(
        model,
        MaterialInput(strain=0.1),
    )

    assert FailureMode.TEARING in response.failure_modes


def test_detect_failure_modes_without_failure() -> None:
    model = make_model()

    modes = detect_failure_modes(
        model,
        MaterialInput(strain=0.01),
        stress=1.0,
        state=MaterialState(),
    )

    assert modes == ()


# ---------------------------------------------------------------------------
# Material plane factors and traces
# ---------------------------------------------------------------------------


def test_material_plane_factors_are_read_only() -> None:
    model = make_model(
        state=MaterialState(
            integrity=0.9,
            damage=0.2,
            fatigue=0.3,
        )
    )

    factors = material_plane_factors(model)

    assert isinstance(factors, MappingProxyType)
    assert factors["material.integrity"] == pytest.approx(0.9)
    assert factors["material.damage_gate"] == pytest.approx(0.8)
    assert factors["material.fatigue_gate"] == pytest.approx(0.7)


def test_active_material_exposes_activation_factor() -> None:
    model = make_model(
        family=MaterialFamily.MUSCLE,
        active_material=True,
        state=MaterialState(activation=0.6),
    )

    factors = material_plane_factors(model)

    assert factors["material.activation"] == pytest.approx(0.6)


def test_passive_material_activation_factor_is_neutral() -> None:
    factors = material_plane_factors(make_model())

    assert factors["material.activation"] == pytest.approx(1.0)


def test_response_to_damage_trace() -> None:
    model = make_model(
        parameters=make_parameters(
            damage_rate=0.5,
            reference_strain=0.1,
        ),
        laws=(
            ConstitutiveLaw(
                law_id="damage",
                kind=ConstitutiveLawKind.DAMAGE,
            ),
        ),
    )
    response = evaluate_material(
        model,
        MaterialInput(strain=0.2),
    )

    trace = response_to_trace(
        response,
        trace_id="damage_trace",
        kind=ConstitutiveLawKind.DAMAGE,
        effect=MaterialEffect.HARMFUL,
    )

    assert isinstance(trace, MaterialTrace)
    assert trace.magnitude > 0.0
    assert trace.effect is MaterialEffect.HARMFUL
    assert isinstance(trace.metadata, MappingProxyType)


def test_response_to_creep_trace() -> None:
    model = make_model(
        parameters=make_parameters(
            elastic_modulus=100.0,
            relaxed_modulus=50.0,
            creep_time_constant=1.0,
        ),
        laws=(
            ConstitutiveLaw(
                law_id="sls",
                kind=ConstitutiveLawKind.STANDARD_LINEAR_SOLID,
            ),
        ),
    )
    response = evaluate_material(
        model,
        MaterialInput(strain=0.1),
    )

    trace = response_to_trace(
        response,
        trace_id="creep_trace",
        kind=ConstitutiveLawKind.STANDARD_LINEAR_SOLID,
    )

    assert trace.magnitude > 0.0


# ---------------------------------------------------------------------------
# Snapshot adaptation
# ---------------------------------------------------------------------------


def test_material_state_from_core_style_snapshot() -> None:
    snapshot = {
        "damage": 0.2,
        "fatigue": 0.1,
        "remodeling": 0.3,
        "activation": 0.4,
        "history": 5.0,
        "energy": 2.0,
        "strain": 0.2,
        "creep_strain": 0.05,
        "residual_strain": 0.01,
        "rheology_time": 3.0,
        "dissipated_energy": 1.0,
        "failed": False,
        "total_force_component": 7.0,
        "length_velocity": 0.2,
        "custom_field": "preserved",
    }

    state = material_state_from_snapshot(snapshot)

    assert state.damage == pytest.approx(0.2)
    assert state.fatigue == pytest.approx(0.1)
    assert state.elastic_strain == pytest.approx(0.15)
    assert state.time == pytest.approx(3.0)
    assert state.metadata["custom_field"] == "preserved"


def test_failed_snapshot_maps_to_failure_state() -> None:
    state = material_state_from_snapshot(
        {
            "failed": True,
            "damage": 1.0,
        }
    )

    assert state.failed is True
    assert state.phase is MaterialPhase.FAILED
    assert state.failure_mode is FailureMode.BIOLOGICAL_FAILURE


def test_snapshot_adapter_rejects_non_mapping() -> None:
    with pytest.raises(ROIFMaterialError):
        material_state_from_snapshot([])  # type: ignore[arg-type]


def test_model_from_core_material() -> None:
    core = FakeCoreMaterial(
        {
            "name": "core_material",
            "parameters": {
                "stiffness": 200.0,
                "damping": 3.0,
                "density": 1200.0,
                "fatigue_rate": 0.1,
                "damage_rate": 0.2,
                "reference_force": 10.0,
            },
            "damage": 0.1,
            "fatigue": 0.2,
            "activation": 0.5,
            "strain": 0.1,
        }
    )

    model = model_from_core_material(
        core,
        material_id="adapted",
        family=MaterialFamily.MUSCLE,
    )

    assert model.descriptor.material_id == "adapted"
    assert model.descriptor.family is MaterialFamily.MUSCLE
    assert model.descriptor.biological is True
    assert model.parameters.elastic_modulus == pytest.approx(200.0)
    assert model.parameters.damping == pytest.approx(3.0)
    assert model.state.damage == pytest.approx(0.1)
    assert model.metadata["source"] == "core.material.Material.snapshot"


def test_model_from_core_material_infers_sls_when_relaxed_stiffness_exists() -> None:
    core = FakeCoreMaterial(
        {
            "parameters": {
                "stiffness": 100.0,
                "relaxed_stiffness": 50.0,
            }
        }
    )

    model = model_from_core_material(
        core,
        material_id="sls_material",
    )

    kinds = {law.kind for law in model.laws}

    assert ConstitutiveLawKind.LINEAR_ELASTIC in kinds
    assert ConstitutiveLawKind.STANDARD_LINEAR_SOLID in kinds


def test_model_from_core_material_requires_snapshot_provider() -> None:
    with pytest.raises(TypeError):
        model_from_core_material(
            object(),  # type: ignore[arg-type]
            material_id="bad",
        )


# ---------------------------------------------------------------------------
# Factory and lookup
# ---------------------------------------------------------------------------


def test_default_material_model() -> None:
    model = default_material_model(
        "concrete",
        family=MaterialFamily.CONCRETE,
    )

    assert model.descriptor.material_id == "concrete"
    assert model.descriptor.family is MaterialFamily.CONCRETE
    assert model.laws[0].kind is ConstitutiveLawKind.LINEAR_ELASTIC


def test_material_models_by_id_returns_read_only_lookup() -> None:
    steel = default_material_model(
        "steel",
        family=MaterialFamily.METAL,
    )
    timber = default_material_model(
        "timber",
        family=MaterialFamily.TIMBER,
    )

    lookup = material_models_by_id((steel, timber))

    assert isinstance(lookup, MappingProxyType)
    assert set(lookup) == {"steel", "timber"}


def test_material_models_by_id_rejects_duplicates() -> None:
    model = default_material_model("same")

    with pytest.raises(ROIFMaterialError):
        material_models_by_id((model, model))


# ---------------------------------------------------------------------------
# Low-level response objects
# ---------------------------------------------------------------------------


def test_law_contribution_rejects_negative_dissipation() -> None:
    with pytest.raises(ROIFMaterialError):
        LawContribution(
            law_id="law",
            kind=ConstitutiveLawKind.VISCOELASTIC,
            stress=1.0,
            tangent_modulus=1.0,
            dissipated_energy=-1.0,
        )


def test_material_trace_rejects_invalid_retention() -> None:
    with pytest.raises(ROIFMaterialError):
        MaterialTrace(
            trace_id="trace",
            material_id="material",
            time=0.0,
            kind=ConstitutiveLawKind.CREEP,
            magnitude=0.1,
            retained_fraction=1.1,
            reversibility=0.0,
        )
