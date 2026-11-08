"""Validate square or rectangular local patch merge sizes."""

from collections.abc import Sequence


SpatialMergeSize = int | Sequence[int]


def spatial_merge_shape(size: SpatialMergeSize, name: str = "spatial_merge_size") -> tuple[int, int]:
    """An integer keeps its original square meaning; a pair is [height, width]."""
    if isinstance(size, int) and not isinstance(size, bool):
        shape = (size, size)
    elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 2:
        shape = (size[0], size[1])
    else:
        raise ValueError(f"{name} must be a positive integer or [height, width]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError(f"{name} must contain positive integers")
    return shape
