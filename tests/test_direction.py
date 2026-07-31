"""
Tests for roif.direction.

The suite fixes the invariants of the ROIF load-direction model:

    d = normalize(v)
    f = magnitude · d

Direction stores orientation and load magnitude separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from roif.direction import (
    Direction,
    DirectionError,
)


# ---------------------------------------------------------------------------
# Construction and normalization
# ---------------------------------------------------------------------------


def test_direction_normalizes_input_vector() -> None:
    """Input vector must be normalized automatically."""

    direction = Direction(
        vector=[10.0, 0.0, 0.0],
    )

    assert np.allclose(
        direction.unit_vector,
        [1.0, 0.0, 0.0],
    )
    assert np.linalg.norm(direction.unit_vector) == pytest.approx(1.0)


def test_direction_preserves_dimension() -> None:
    """Direction dimension must match the input vector dimension."""

    direction_2d = Direction(
        vector=[1.0, 1.0],
    )
    direction_3d = Direction(
        vector=[1.0, 1.0, 1.0],
    )

    assert direction_2d.dimension == 2
    assert direction_3d.dimension == 3


def test_default_magnitude_is_one() -> None:
    """Magnitude must equal one when omitted."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.magnitude == pytest.approx(1.0)


def test_name_and_source_are_trimmed() -> None:
    """Readable identifiers must be stripped of outer whitespace."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        name="  gravity  ",
        source="  body weight  ",
    )

    assert direction.name == "gravity"
    assert direction.source == "body weight"


# ---------------------------------------------------------------------------
# Force vector
# ---------------------------------------------------------------------------


def test_force_vector_equals_magnitude_times_unit_vector() -> None:
    """Complete load vector must be f = magnitude · d."""

    direction = Direction(
        vector=[3.0, 4.0, 0.0],
        magnitude=10.0,
    )

    assert np.allclose(
        direction.unit_vector,
        [0.6, 0.8, 0.0],
    )
    assert np.allclose(
        direction.force_vector,
        [6.0, 8.0, 0.0],
    )


def test_force_vector_is_independent_copy() -> None:
    """Mutating a returned force vector must not alter Direction."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=10.0,
    )

    force = direction.force_vector
    force[0] = 999.0

    assert np.allclose(
        direction.force_vector,
        [10.0, 0.0, 0.0],
    )


def test_unit_vector_is_defensive_copy() -> None:
    """Mutating unit_vector output must not change internal orientation."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    vector = direction.unit_vector
    vector[0] = 0.0

    assert np.allclose(
        direction.unit_vector,
        [1.0, 0.0, 0.0],
    )


def test_zero_magnitude_is_valid() -> None:
    """Zero load may preserve a known orientation."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=0.0,
    )

    assert direction.is_zero_load
    assert np.allclose(
        direction.force_vector,
        [0.0, 0.0, 0.0],
    )


def test_positive_magnitude_is_not_zero_load() -> None:
    """A non-zero magnitude must not be classified as zero load."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=1.0,
    )

    assert not direction.is_zero_load


# ---------------------------------------------------------------------------
# Dot products and angles
# ---------------------------------------------------------------------------


def test_dot_with_parallel_direction_is_one() -> None:
    """Parallel unit directions must have dot product one."""

    first = Direction(
        vector=[1.0, 0.0, 0.0],
    )
    second = Direction(
        vector=[5.0, 0.0, 0.0],
        magnitude=100.0,
    )

    assert first.dot(second) == pytest.approx(1.0)


def test_dot_ignores_magnitude() -> None:
    """Directional dot product must compare orientations only."""

    first = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=2.0,
    )
    second = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=1000.0,
    )

    assert first.dot(second) == pytest.approx(1.0)


def test_dot_with_orthogonal_vector_is_zero() -> None:
    """Orthogonal directions must have zero dot product."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.dot([0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_dot_with_opposite_direction_is_minus_one() -> None:
    """Opposite directions must have dot product minus one."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.dot([-1.0, 0.0, 0.0]) == pytest.approx(-1.0)


def test_angle_to_parallel_direction_is_zero() -> None:
    """Parallel directions must have zero angular separation."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.angle_to(
        [10.0, 0.0, 0.0],
    ) == pytest.approx(0.0)


def test_angle_to_orthogonal_direction_is_pi_over_two() -> None:
    """Orthogonal directions must be separated by π/2."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.angle_to(
        [0.0, 1.0, 0.0],
    ) == pytest.approx(np.pi / 2.0)


def test_angle_to_opposite_direction_is_pi() -> None:
    """Opposite directions must be separated by π."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.angle_to(
        [-1.0, 0.0, 0.0],
    ) == pytest.approx(np.pi)


def test_angle_can_be_returned_in_degrees() -> None:
    """Angle conversion to degrees must be supported."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.angle_to(
        [0.0, 1.0, 0.0],
        degrees=True,
    ) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def test_positive_projection_accepts_forward_axis() -> None:
    """Positive projection must preserve forward alignment."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.positive_projection_on(
        [1.0, 0.0, 0.0],
    ) == pytest.approx(1.0)


def test_positive_projection_rejects_opposite_axis() -> None:
    """Positive-part projection must reject negative alignment."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.positive_projection_on(
        [-1.0, 0.0, 0.0],
    ) == pytest.approx(0.0)


def test_negative_projection_accepts_opposite_axis() -> None:
    """Compression-oriented projection must accept negative alignment."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.negative_projection_on(
        [-1.0, 0.0, 0.0],
    ) == pytest.approx(1.0)


def test_negative_projection_rejects_forward_axis() -> None:
    """Compression-oriented projection must reject positive alignment."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.negative_projection_on(
        [1.0, 0.0, 0.0],
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ([1.0, 0.0, 0.0], 1.0),
        ([-1.0, 0.0, 0.0], 1.0),
        ([0.0, 1.0, 0.0], 0.0),
    ],
)
def test_absolute_projection_is_bidirectional(
    axis: list[float],
    expected: float,
) -> None:
    """Absolute projection must serve both axial signs."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    assert direction.absolute_projection_on(axis) == pytest.approx(expected)


def test_diagonal_projection_equals_cosine() -> None:
    """Projection onto a 45-degree axis must equal 1/sqrt(2)."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    result = direction.positive_projection_on(
        [1.0, 1.0, 0.0],
    )

    assert result == pytest.approx(1.0 / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Transformations and copies
# ---------------------------------------------------------------------------


def test_scaled_changes_magnitude_only() -> None:
    """Scaling must preserve orientation and multiply magnitude."""

    direction = Direction(
        vector=[1.0, 1.0, 0.0],
        magnitude=10.0,
        name="load",
        source="test",
        metadata={"case": 1},
    )

    scaled = direction.scaled(2.5)

    assert scaled.magnitude == pytest.approx(25.0)
    assert np.allclose(
        scaled.unit_vector,
        direction.unit_vector,
    )
    assert scaled.name == direction.name
    assert scaled.source == direction.source
    assert scaled.metadata_dict == direction.metadata_dict


def test_scaled_does_not_mutate_original() -> None:
    """scaled() must return an independent Direction."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=10.0,
    )

    scaled = direction.scaled(3.0)

    assert direction.magnitude == pytest.approx(10.0)
    assert scaled.magnitude == pytest.approx(30.0)
    assert scaled is not direction


def test_with_magnitude_replaces_magnitude() -> None:
    """with_magnitude() must preserve orientation and replace load size."""

    direction = Direction(
        vector=[0.0, 1.0, 0.0],
        magnitude=10.0,
    )

    changed = direction.with_magnitude(250.0)

    assert changed.magnitude == pytest.approx(250.0)
    assert np.allclose(
        changed.unit_vector,
        [0.0, 1.0, 0.0],
    )
    assert direction.magnitude == pytest.approx(10.0)


def test_reversed_flips_orientation_only() -> None:
    """reversed() must invert d while preserving magnitude."""

    direction = Direction(
        vector=[1.0, -2.0, 3.0],
        magnitude=25.0,
        name="load",
        source="source",
        metadata={"id": 42},
    )

    reversed_direction = direction.reversed()

    assert np.allclose(
        reversed_direction.unit_vector,
        -direction.unit_vector,
    )
    assert reversed_direction.magnitude == pytest.approx(
        direction.magnitude
    )
    assert reversed_direction.name == direction.name
    assert reversed_direction.source == direction.source
    assert reversed_direction.metadata_dict == direction.metadata_dict


def test_rotated_applies_configuration_matrix() -> None:
    """Rotation matrix must transform the direction orientation."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=100.0,
    )

    rotation_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rotated = direction.rotated(rotation_z_90)

    assert np.allclose(
        rotated.unit_vector,
        [0.0, 1.0, 0.0],
    )
    assert rotated.magnitude == pytest.approx(100.0)


def test_rotated_normalizes_result() -> None:
    """
    Slightly scaled transformation matrices must not change direction length.
    """

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    configuration = np.array(
        [
            [0.0, -2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    rotated = direction.rotated(configuration)

    assert np.linalg.norm(rotated.unit_vector) == pytest.approx(1.0)
    assert np.allclose(
        rotated.unit_vector,
        [0.0, 1.0, 0.0],
    )


def test_identity_rotation_preserves_direction() -> None:
    """Identity configuration must preserve orientation."""

    direction = Direction(
        vector=[1.0, 2.0, 3.0],
        magnitude=10.0,
    )

    rotated = direction.rotated(np.eye(3))

    assert np.allclose(
        rotated.unit_vector,
        direction.unit_vector,
    )
    assert rotated.magnitude == pytest.approx(direction.magnitude)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_is_copied_during_construction() -> None:
    """External dictionary changes must not alter stored metadata."""

    metadata = {
        "patient": "case_1",
        "phase": "standing",
    }

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        metadata=metadata,
    )

    metadata["phase"] = "walking"

    assert direction.metadata_dict["phase"] == "standing"


def test_metadata_dict_is_defensive_copy() -> None:
    """Mutating metadata_dict output must not alter Direction."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
        metadata={"phase": "standing"},
    )

    metadata = direction.metadata_dict
    metadata["phase"] = "walking"

    assert direction.metadata_dict["phase"] == "standing"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_as_dict_returns_serializable_structure() -> None:
    """as_dict() must expose vector, magnitude and descriptors."""

    direction = Direction(
        vector=[0.0, 0.0, -2.0],
        magnitude=700.0,
        name="gravity",
        source="body_weight",
        metadata={"configuration": "standing"},
    )

    data = direction.as_dict()

    assert data == {
        "vector": [0.0, 0.0, -1.0],
        "magnitude": 700.0,
        "name": "gravity",
        "source": "body_weight",
        "metadata": {
            "configuration": "standing",
        },
    }


def test_from_dict_restores_direction() -> None:
    """Serialized data must reconstruct an equivalent Direction."""

    data = {
        "vector": [3.0, 4.0, 0.0],
        "magnitude": 25.0,
        "name": "resultant",
        "source": "combined",
        "metadata": {
            "case": 7,
        },
    }

    direction = Direction.from_dict(data)

    assert np.allclose(
        direction.unit_vector,
        [0.6, 0.8, 0.0],
    )
    assert direction.magnitude == pytest.approx(25.0)
    assert direction.name == "resultant"
    assert direction.source == "combined"
    assert direction.metadata_dict == {"case": 7}


def test_serialization_round_trip() -> None:
    """as_dict() followed by from_dict() must preserve the object state."""

    original = Direction(
        vector=[1.0, -2.0, 3.0],
        magnitude=150.0,
        name="muscle_pull",
        source="psoas",
        metadata={
            "active": True,
            "configuration": "gait",
        },
    )

    restored = Direction.from_dict(original.as_dict())

    assert np.allclose(
        restored.unit_vector,
        original.unit_vector,
    )
    assert restored.magnitude == pytest.approx(original.magnitude)
    assert restored.name == original.name
    assert restored.source == original.source
    assert restored.metadata_dict == original.metadata_dict


# ---------------------------------------------------------------------------
# Construction from complete force vector
# ---------------------------------------------------------------------------


def test_from_force_vector_extracts_magnitude_and_orientation() -> None:
    """Force-vector norm must become magnitude."""

    direction = Direction.from_force_vector(
        [3.0, 4.0, 0.0],
        name="force",
    )

    assert direction.magnitude == pytest.approx(5.0)
    assert np.allclose(
        direction.unit_vector,
        [0.6, 0.8, 0.0],
    )
    assert np.allclose(
        direction.force_vector,
        [3.0, 4.0, 0.0],
    )


def test_from_force_vector_preserves_descriptors() -> None:
    """Optional name, source and metadata must be preserved."""

    direction = Direction.from_force_vector(
        [0.0, 0.0, -700.0],
        name="gravity",
        source="body_weight",
        metadata={"phase": "standing"},
    )

    assert direction.name == "gravity"
    assert direction.source == "body_weight"
    assert direction.metadata_dict == {"phase": "standing"}


# ---------------------------------------------------------------------------
# Combining loads
# ---------------------------------------------------------------------------


def test_combine_sums_force_vectors() -> None:
    """Resultant load must equal the vector sum of all loads."""

    horizontal = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=100.0,
    )
    vertical = Direction(
        vector=[0.0, 0.0, -1.0],
        magnitude=700.0,
    )

    resultant = Direction.combine(
        [horizontal, vertical],
        name="resultant",
    )

    assert np.allclose(
        resultant.force_vector,
        [100.0, 0.0, -700.0],
    )
    assert resultant.name == "resultant"


def test_combine_parallel_directions_adds_magnitudes() -> None:
    """Parallel forces must produce their summed magnitude."""

    first = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=100.0,
    )
    second = Direction(
        vector=[5.0, 0.0, 0.0],
        magnitude=250.0,
    )

    resultant = Direction.combine([first, second])

    assert resultant.magnitude == pytest.approx(350.0)
    assert np.allclose(
        resultant.unit_vector,
        [1.0, 0.0, 0.0],
    )


def test_combine_preserves_result_descriptors() -> None:
    """Descriptors supplied for the resultant must be stored."""

    first = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=100.0,
    )
    second = Direction(
        vector=[0.0, 1.0, 0.0],
        magnitude=100.0,
    )

    resultant = Direction.combine(
        [first, second],
        name="combined_load",
        source="fast_layer",
        metadata={"step": 10},
    )

    assert resultant.name == "combined_load"
    assert resultant.source == "fast_layer"
    assert resultant.metadata_dict == {"step": 10}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "magnitude",
    [-1.0, -0.001, np.inf, -np.inf, np.nan],
)
def test_invalid_magnitude_is_rejected(
    magnitude: float,
) -> None:
    """Magnitude must be finite and non-negative."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[1.0, 0.0, 0.0],
            magnitude=magnitude,
        )


@pytest.mark.parametrize(
    "factor",
    [-1.0, np.inf, -np.inf, np.nan],
)
def test_invalid_scale_factor_is_rejected(
    factor: float,
) -> None:
    """Scale factor must be finite and non-negative."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction.scaled(factor)


def test_zero_vector_is_rejected() -> None:
    """An orientation cannot be defined by a zero vector."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[0.0, 0.0, 0.0],
        )


def test_empty_vector_is_rejected() -> None:
    """Direction vector cannot be empty."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[],
        )


def test_multidimensional_vector_is_rejected() -> None:
    """Direction input must be one-dimensional."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )


@pytest.mark.parametrize(
    "vector",
    [
        [np.nan, 0.0, 0.0],
        [np.inf, 0.0, 0.0],
        [-np.inf, 0.0, 0.0],
    ],
)
def test_non_finite_vector_is_rejected(
    vector: list[float],
) -> None:
    """Direction components must be finite."""

    with pytest.raises(DirectionError):
        Direction(
            vector=vector,
        )


def test_empty_name_is_rejected() -> None:
    """Empty or whitespace-only names must be rejected."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[1.0, 0.0, 0.0],
            name="   ",
        )


def test_empty_source_is_rejected() -> None:
    """Empty or whitespace-only sources must be rejected."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[1.0, 0.0, 0.0],
            source="   ",
        )


def test_non_string_name_is_rejected() -> None:
    """Name must be either a string or None."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[1.0, 0.0, 0.0],
            name=123,  # type: ignore[arg-type]
        )


def test_non_string_source_is_rejected() -> None:
    """Source must be either a string or None."""

    with pytest.raises(DirectionError):
        Direction(
            vector=[1.0, 0.0, 0.0],
            source=123,  # type: ignore[arg-type]
        )


def test_dot_dimension_mismatch_is_rejected() -> None:
    """Compared vectors must have the same dimension."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction.dot([1.0, 0.0])


def test_dot_zero_vector_is_rejected() -> None:
    """A zero comparison vector has no orientation."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction.dot([0.0, 0.0, 0.0])


def test_direction_object_dimension_mismatch_is_rejected() -> None:
    """Direction objects of different dimensions cannot be compared."""

    direction_3d = Direction(
        vector=[1.0, 0.0, 0.0],
    )
    direction_2d = Direction(
        vector=[1.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction_3d.dot(direction_2d)


def test_invalid_rotation_shape_is_rejected() -> None:
    """Configuration matrix must have shape n × n."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction.rotated(np.eye(2))


def test_non_finite_rotation_is_rejected() -> None:
    """Configuration matrix must contain finite values."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    matrix = np.eye(3)
    matrix[0, 0] = np.nan

    with pytest.raises(DirectionError):
        direction.rotated(matrix)


def test_rotation_collapsing_direction_is_rejected() -> None:
    """A transformation cannot collapse orientation into a zero vector."""

    direction = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        direction.rotated(np.zeros((3, 3)))


def test_from_force_vector_rejects_zero_force() -> None:
    """
    A zero force vector cannot define orientation when constructing a new
    Direction.
    """

    with pytest.raises(DirectionError):
        Direction.from_force_vector(
            [0.0, 0.0, 0.0],
        )


def test_from_force_vector_rejects_empty_vector() -> None:
    """Force vector cannot be empty."""

    with pytest.raises(DirectionError):
        Direction.from_force_vector([])


def test_from_force_vector_rejects_multidimensional_input() -> None:
    """Force vector must be one-dimensional."""

    with pytest.raises(DirectionError):
        Direction.from_force_vector(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )


def test_from_force_vector_rejects_non_finite_values() -> None:
    """Force vector must contain finite values."""

    with pytest.raises(DirectionError):
        Direction.from_force_vector(
            [np.inf, 0.0, 0.0],
        )


def test_from_dict_requires_vector() -> None:
    """Serialized mapping must contain the vector field."""

    with pytest.raises(DirectionError):
        Direction.from_dict(
            {
                "magnitude": 10.0,
            }
        )


def test_combine_rejects_empty_sequence() -> None:
    """At least one direction is required."""

    with pytest.raises(DirectionError):
        Direction.combine([])


def test_combine_rejects_non_direction_objects() -> None:
    """combine() must accept Direction instances only."""

    valid = Direction(
        vector=[1.0, 0.0, 0.0],
    )

    with pytest.raises(DirectionError):
        Direction.combine(
            [
                valid,
                [0.0, 1.0, 0.0],  # type: ignore[list-item]
            ]
        )


def test_combine_rejects_dimension_mismatch() -> None:
    """All combined loads must have equal dimensions."""

    direction_3d = Direction(
        vector=[1.0, 0.0, 0.0],
    )
    direction_2d = Direction(
        vector=[1.0, 0.0],
    )

    with pytest.raises(DirectionError):
        Direction.combine(
            [
                direction_3d,
                direction_2d,
            ]
        )


def test_combine_rejects_zero_resultant() -> None:
    """Exactly cancelling forces have no resultant orientation."""

    first = Direction(
        vector=[1.0, 0.0, 0.0],
        magnitude=100.0,
    )
    second = Direction(
        vector=[-1.0, 0.0, 0.0],
        magnitude=100.0,
    )

    with pytest.raises(DirectionError):
        Direction.combine(
            [
                first,
                second,
            ]
        )