"""
ROIF load-direction model.

This module introduces a validated representation of an external or internal
load direction used by the directed-capacity model κ(d, x) and the fast layer

    A(x) T = f.

A Direction stores:

    d          — normalized direction vector;
    magnitude  — scalar load magnitude;
    source     — optional physical or functional source;
    name       — readable identifier;
    metadata   — optional descriptive information.

The normalized vector and load magnitude are stored separately so that:

    direction.vector

describes orientation, while

    direction.force_vector

returns the full vector load

    f = magnitude · d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]

_EPSILON = 1e-12


class DirectionError(ValueError):
    """Raised when invalid direction data is supplied."""


@dataclass(frozen=True, slots=True)
class Direction:
    """
    Validated load direction used by the ROIF analytical layer.

    Parameters
    ----------
    vector:
        Direction vector. It is normalized automatically.

    magnitude:
        Non-negative scalar load magnitude.

    name:
        Optional readable name, for example ``gravity`` or ``muscle_pull``.

    source:
        Optional physical, anatomical, environmental, or control source.

    metadata:
        Optional descriptive values. A defensive dictionary copy is stored.

    Notes
    -----
    ``vector`` represents orientation only.

    The complete load vector is available through:

        force_vector = magnitude * vector
    """

    vector: Iterable[float] | FloatArray
    magnitude: float = 1.0
    name: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    _unit_vector: FloatArray = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        unit_vector = self._normalize_vector(
            self.vector,
            name="vector",
        )
        magnitude = self._validate_non_negative_scalar(
            self.magnitude,
            name="magnitude",
        )
        name = self._validate_optional_text(
            self.name,
            name="name",
        )
        source = self._validate_optional_text(
            self.source,
            name="source",
        )

        metadata = (
            {}
            if self.metadata is None
            else dict(self.metadata)
        )

        object.__setattr__(self, "_unit_vector", unit_vector)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_metadata", metadata)

    @property
    def dimension(self) -> int:
        """Return spatial dimension of the direction vector."""

        return int(self._unit_vector.size)

    @property
    def unit_vector(self) -> FloatArray:
        """Return a defensive copy of normalized direction d."""

        return self._unit_vector.copy()

    @property
    def force_vector(self) -> FloatArray:
        """
        Return complete load vector f.

        The result is:

            f = magnitude · d
        """

        return self.magnitude * self._unit_vector

    @property
    def metadata_dict(self) -> dict[str, Any]:
        """Return a defensive copy of metadata."""

        return dict(self._metadata)

    @property
    def is_zero_load(self) -> bool:
        """Return True when magnitude is numerically zero."""

        return self.magnitude <= _EPSILON

    def dot(
        self,
        other: Direction | Iterable[float] | FloatArray,
    ) -> float:
        """
        Return directional dot product.

        For another Direction, only unit vectors are compared.

        For an array-like input, the vector is normalized first.
        """

        other_vector = self._coerce_unit_vector(
            other,
            expected_dimension=self.dimension,
        )

        return float(np.dot(self._unit_vector, other_vector))

    def angle_to(
        self,
        other: Direction | Iterable[float] | FloatArray,
        *,
        degrees: bool = False,
    ) -> float:
        """
        Return the smallest angle to another direction.

        Parameters
        ----------
        other:
            Another Direction or vector.

        degrees:
            When True, return degrees instead of radians.
        """

        cosine = np.clip(self.dot(other), -1.0, 1.0)
        angle = float(np.arccos(cosine))

        if degrees:
            return float(np.degrees(angle))

        return angle

    def positive_projection_on(
        self,
        axis: Direction | Iterable[float] | FloatArray,
    ) -> float:
        """
        Return positive-part directional projection.

        This computes:

            <d, a>_+ = max(0, <d, a>)
        """

        return max(0.0, self.dot(axis))

    def negative_projection_on(
        self,
        axis: Direction | Iterable[float] | FloatArray,
    ) -> float:
        """
        Return compression-oriented projection.

        This computes:

            max(0, -<d, a>)
        """

        return max(0.0, -self.dot(axis))

    def absolute_projection_on(
        self,
        axis: Direction | Iterable[float] | FloatArray,
    ) -> float:
        """
        Return bidirectional axial projection.

        This computes:

            |<d, a>|
        """

        return abs(self.dot(axis))

    def scaled(self, factor: float) -> Direction:
        """
        Return an independent direction with scaled load magnitude.

        Orientation is unchanged.
        """

        factor_value = self._validate_non_negative_scalar(
            factor,
            name="factor",
        )

        return Direction(
            vector=self._unit_vector,
            magnitude=self.magnitude * factor_value,
            name=self.name,
            source=self.source,
            metadata=self._metadata,
        )

    def with_magnitude(self, magnitude: float) -> Direction:
        """Return an independent copy with another magnitude."""

        return Direction(
            vector=self._unit_vector,
            magnitude=magnitude,
            name=self.name,
            source=self.source,
            metadata=self._metadata,
        )

    def rotated(
        self,
        configuration: Iterable[Iterable[float]] | FloatArray,
    ) -> Direction:
        """
        Return the direction transformed by a configuration matrix.

        The transformation is:

            d(x) = normalize(R(x) @ d)

        The load magnitude is preserved.
        """

        matrix = np.asarray(configuration, dtype=np.float64)

        expected_shape = (self.dimension, self.dimension)
        if matrix.shape != expected_shape:
            raise DirectionError(
                "configuration must have shape "
                f"{expected_shape}, received {matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise DirectionError(
                "configuration contains non-finite values."
            )

        rotated_vector = matrix @ self._unit_vector

        return Direction(
            vector=rotated_vector,
            magnitude=self.magnitude,
            name=self.name,
            source=self.source,
            metadata=self._metadata,
        )

    def reversed(self) -> Direction:
        """
        Return a direction with opposite orientation.

        Magnitude and metadata are preserved.
        """

        return Direction(
            vector=-self._unit_vector,
            magnitude=self.magnitude,
            name=self.name,
            source=self.source,
            metadata=self._metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "vector": self._unit_vector.tolist(),
            "magnitude": self.magnitude,
            "name": self.name,
            "source": self.source,
            "metadata": dict(self._metadata),
        }

    @classmethod
    def from_force_vector(
        cls,
        force_vector: Iterable[float] | FloatArray,
        *,
        name: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Direction:
        """
        Construct Direction from a complete load vector f.

        Its norm becomes magnitude and its normalized orientation becomes d.
        """

        force = np.asarray(force_vector, dtype=np.float64)

        if force.ndim != 1:
            raise DirectionError(
                "force_vector must be a one-dimensional vector."
            )

        if force.size == 0:
            raise DirectionError(
                "force_vector cannot be empty."
            )

        if not np.all(np.isfinite(force)):
            raise DirectionError(
                "force_vector contains non-finite values."
            )

        magnitude = float(np.linalg.norm(force))

        if magnitude <= _EPSILON:
            raise DirectionError(
                "force_vector cannot be a zero vector."
            )

        return cls(
            vector=force,
            magnitude=magnitude,
            name=name,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Direction:
        """Construct Direction from a serialized mapping."""

        if "vector" not in data:
            raise DirectionError(
                "Direction mapping must contain 'vector'."
            )

        return cls(
            vector=data["vector"],
            magnitude=data.get("magnitude", 1.0),
            name=data.get("name"),
            source=data.get("source"),
            metadata=data.get("metadata"),
        )

    @staticmethod
    def combine(
        directions: Iterable[Direction],
        *,
        name: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Direction:
        """
        Combine several loads by vector summation.

        Each load contributes:

            f_i = magnitude_i · d_i

        The resulting direction is constructed from:

            f_total = Σ f_i

        All directions must have the same dimension.
        """

        direction_list = list(directions)

        if not direction_list:
            raise DirectionError(
                "At least one direction is required for combination."
            )

        dimension = direction_list[0].dimension

        for direction in direction_list:
            if not isinstance(direction, Direction):
                raise DirectionError(
                    "combine accepts Direction objects only."
                )

            if direction.dimension != dimension:
                raise DirectionError(
                    "All directions must have the same dimension."
                )

        total_force = np.sum(
            [direction.force_vector for direction in direction_list],
            axis=0,
            dtype=np.float64,
        )

        if float(np.linalg.norm(total_force)) <= _EPSILON:
            raise DirectionError(
                "Combined force vector is zero; resultant direction "
                "is undefined."
            )

        return Direction.from_force_vector(
            total_force,
            name=name,
            source=source,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_unit_vector(
        value: Direction | Iterable[float] | FloatArray,
        *,
        expected_dimension: int,
    ) -> FloatArray:
        if isinstance(value, Direction):
            if value.dimension != expected_dimension:
                raise DirectionError(
                    "Direction dimensions do not match."
                )

            return value._unit_vector

        return Direction._normalize_vector(
            value,
            name="other",
            expected_dimension=expected_dimension,
        )

    @staticmethod
    def _normalize_vector(
        vector: Iterable[float] | FloatArray,
        *,
        name: str,
        expected_dimension: int | None = None,
    ) -> FloatArray:
        result = np.asarray(vector, dtype=np.float64)

        if result.ndim != 1:
            raise DirectionError(
                f"{name} must be a one-dimensional vector."
            )

        if expected_dimension is not None and result.size != expected_dimension:
            raise DirectionError(
                f"{name} must have dimension {expected_dimension}, "
                f"received {result.size}."
            )

        if result.size == 0:
            raise DirectionError(
                f"{name} cannot be empty."
            )

        if not np.all(np.isfinite(result)):
            raise DirectionError(
                f"{name} contains non-finite values."
            )

        norm = float(np.linalg.norm(result))

        if norm <= _EPSILON:
            raise DirectionError(
                f"{name} cannot be a zero vector."
            )

        return result / norm

    @staticmethod
    def _validate_non_negative_scalar(
        value: float,
        *,
        name: str,
    ) -> float:
        result = float(value)

        if not np.isfinite(result):
            raise DirectionError(
                f"{name} must be finite."
            )

        if result < 0.0:
            raise DirectionError(
                f"{name} must be non-negative, received {result}."
            )

        return result

    @staticmethod
    def _validate_optional_text(
        value: str | None,
        *,
        name: str,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise DirectionError(
                f"{name} must be a string or None."
            )

        normalized = value.strip()

        if not normalized:
            raise DirectionError(
                f"{name} cannot be empty."
            )

        return normalized