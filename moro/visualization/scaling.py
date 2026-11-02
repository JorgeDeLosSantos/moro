"""Dimensionless-independent scale metrics for evaluated robot scenes."""

import math
from collections.abc import Iterable

from .data import SceneData


def resolve_reference_length(scenes: Iterable[SceneData]) -> float | None:
    """Return the largest available characteristic length across scenes.

    Only evaluated DH geometry contributes. Legacy scenes without segments
    are ignored. `None` means no evaluated geometry was available; `0.0`
    means that all evaluated geometry is degenerate. The caller must handle
    the visual fallback separately when the result is zero or unavailable.

    The result is suitable for keeping graphical dimensions constant through
    a precomputed animation sequence.
    """
    lengths = (
        scene.characteristic_length
        for scene in scenes
    )
    available = [length for length in lengths if length is not None]
    if not available:
        return None

    reference_length = max(available)
    if not math.isfinite(reference_length):
        raise ValueError("Reference length must be finite.")
    return reference_length
