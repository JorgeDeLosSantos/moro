"""Backend-independent geometric dimensions for robot visualization.

V3-A resolves physical-looking world-space sizes only. Matplotlib marker-area
conversion, Three.js geometry, framing, and cameras are intentionally handled
by later integration steps.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .data import SceneData
from .scaling import resolve_reference_length


# Nominal ratios relative to the selected reference length.
_FRAME_LENGTH_RATIO = 1.0 / 6.0
_JOINT_RADIUS_RATIO = 0.020
_BASE_RADIUS_RATIO = 0.030
_LINK_RADIUS_RATIO = 0.015

# Visual-only fallback when neither DH geometry nor legacy dimensions
# provide a meaningful scale. It never alters SceneData geometry.
_VISUAL_FALLBACK_LENGTH = 1.0


@dataclass(frozen=True, slots=True)
class VisualizationLayout:
    """Resolved world-space sizes shared by rendering backends.

    All five lengths use the robot's input linear units. The nominal link
    radius is not yet modified by VisualizationStyle.link_linewidth.
    """

    reference_length: float
    frame_length: float
    joint_radius: float
    base_radius: float
    link_radius: float


def _legacy_reference_length(scenes: Sequence[SceneData]) -> float | None:
    """Select the largest finite positive legacy scene dimension, if any."""
    dimensions: list[float] = []

    for scene in scenes:
        try:
            dimension = float(scene.dimension)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(dimension) and dimension > 0.0:
            dimensions.append(dimension)

    return max(dimensions) if dimensions else None


def resolve_layout(
    scenes: Sequence[SceneData],
    *,
    reference_length: float | None = None,
) -> VisualizationLayout:
    """Resolve one stable geometric layout for a scene or animation.

    Scale precedence:
    1. Explicit finite positive `reference_length`.
    2. Maximum positive DH characteristic length across `scenes`.
    3. Maximum finite positive legacy `SceneData.dimension`.
    4. Visual-only fallback length of 1.0.

    `SceneData.characteristic_length` remains unchanged, including when zero
    or unavailable. No camera, plot bounds, or backend-specific sizes are
    calculated here. Resolve once per precomputed animation, not per frame.
    """
    if len(scenes) == 0:
        raise ValueError("scenes must contain at least one scene.")

    if reference_length is not None:
        if isinstance(reference_length, bool):
            raise ValueError("reference_length must be a finite positive number.")
        try:
            length = float(reference_length)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "reference_length must be a finite positive number."
            ) from exc
        if not math.isfinite(length) or length <= 0:
            raise ValueError("reference_length must be a finite positive number.")
    else:
        length = resolve_reference_length(scenes)
        if length is None or length <= 0.0:
            length = _legacy_reference_length(scenes)
        if length is None:
            length = _VISUAL_FALLBACK_LENGTH

    return VisualizationLayout(
        reference_length=length,
        frame_length=_FRAME_LENGTH_RATIO * length,
        joint_radius=_JOINT_RADIUS_RATIO * length,
        base_radius=_BASE_RADIUS_RATIO * length,
        link_radius=_LINK_RADIUS_RATIO * length,
    )
