"""
Public API for the ROIF analytical layer.

This package contains the mathematical and computational components of the
Recursive Organic Integration Framework built above the mechanical core.
"""

from .capacity_tensor import (
    CapacityEvaluation,
    CapacityTensor,
    CapacityTensorError,
    ElementActivity,
    ResponseMode,
    evaluate_capacity_batch,
)
from .direction import Direction, DirectionError
from .fast_layer import (
    FastLayer,
    FastLayerError,
    FastLayerResult,
    SolveMethod,
    solve_fast_layer,
)

__all__ = [
    "CapacityEvaluation",
    "CapacityTensor",
    "CapacityTensorError",
    "Direction",
    "DirectionError",
    "ElementActivity",
    "FastLayer",
    "FastLayerError",
    "FastLayerResult",
    "ResponseMode",
    "SolveMethod",
    "evaluate_capacity_batch",
    "solve_fast_layer",
]
