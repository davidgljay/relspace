"""Utility functions."""
from typing import Tuple, Union
import numpy as np  # type: ignore

ProperShape = Tuple[int, ...]
Shape = Union[int, ProperShape]


def dims(shapable: Union[np.ndarray, Shape]) -> int:
    """Get number of dimensions for a shape."""
    if isinstance(shapable, int):
        return 1
    if isinstance(shapable, np.ndarray):
        return len(shapable.shape)
    return len(shapable)


def shape_lift(shape: Shape) -> ProperShape:
    """Make a shape a proper shape tuple."""
    if isinstance(shape, int):
        return (shape,)
    return shape
