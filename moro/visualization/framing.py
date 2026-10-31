"""Backend-independent scene bounds and camera framing geometry.

V4-A defines conservative world-space bounds. The results are not camera
objects: rendering, viewport aspect ratio and clipping planes belong to V4-B.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from .data import SceneData
from .layout import VisualizationLayout, resolve_layout
from .joint_symbols import build_joint_symbols
from .joint_solids import build_joint_solids, solid_bounding_radius
from .style import VisualizationStyle


Vector3 = tuple[float, float, float]

_DEFAULT_MARGIN = 0.10
_DEFAULT_MIN_EXTENT_FACTOR = 0.10


def _vector3(value, description: str) -> Vector3:
    """Read one finite world-space position without retaining array aliases."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description} must contain three finite real values.") from exc

    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{description} must contain three finite real values.")
    return tuple(float(value) for value in array)


def _nonnegative(value, description: str) -> float:
    """Validate finite world-space sizes; zero is valid for collapsed geometry."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{description} must be a finite nonnegative number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description} must be a finite nonnegative number.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{description} must be a finite nonnegative number.")
    return number


@dataclass(frozen=True, slots=True)
class SceneBounds:
    """Finite axis-aligned bounds in the robot's world-coordinate units."""

    minimum: Vector3
    maximum: Vector3

    def __post_init__(self):
        minimum = _vector3(self.minimum, "minimum")
        maximum = _vector3(self.maximum, "maximum")
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError("Each minimum coordinate must not exceed maximum.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def center(self) -> Vector3:
        """Center of the box, avoiding overflow from direct endpoint addition."""
        return tuple(
            low / 2.0 + high / 2.0
            for low, high in zip(self.minimum, self.maximum)
        )

    @property
    def size(self) -> Vector3:
        """Nonnegative extents along world X, Y and Z."""
        size = tuple(
            high - low for low, high in zip(self.minimum, self.maximum)
        )
        if not all(math.isfinite(value) for value in size):
            raise ValueError("Scene bounds exceed the finite numeric range.")
        return size


@dataclass(frozen=True, slots=True)
class SceneFraming:
    """World-space framing instructions, independent of projection and viewport."""

    bounds: SceneBounds
    center: Vector3
    half_extent: float
    fit_radius: float


def union_bounds(bounds: Sequence[SceneBounds]) -> SceneBounds:
    """Compute the AABB union of a nonempty sequence of boxes."""
    if not bounds:
        raise ValueError("bounds must contain at least one box.")

    return SceneBounds(
        tuple(min(box.minimum[i] for box in bounds) for i in range(3)),
        tuple(max(box.maximum[i] for box in bounds) for i in range(3)),
    )


def compute_scene_bounds(
    scene: SceneData,
    *,
    layout: VisualizationLayout | None = None,
    style: VisualizationStyle | None = None,
) -> SceneBounds:
    """Bound visible robot primitives in one evaluated configuration.

    Includes DH translation segments, or legacy joint-to-joint links when no
    DH data is present; visible joints and their nominal radii; visible local
    frame origins and axis tips; and the EE position if a trajectory is shown.

    Link radii are conservative world-space estimates using the Three.js
    linewidth ratio. Explicit Matplotlib marker sizes are in points-squared,
    so they cannot be interpreted as physical radii here. Global axes and
    grids do not contribute. A fully hidden or empty scene has a zero-size
    box at the world origin, expanded visually by resolve_framing.
    """
    if layout is None:
        layout = resolve_layout([scene])
    if style is None:
        style = VisualizationStyle()

    points: list[Vector3] = []

    def add_point(value, *, radius: float = 0.0, description="position"):
        xyz = _vector3(value, description)
        if radius:
            points.append(tuple(coordinate - radius for coordinate in xyz))
            points.append(tuple(coordinate + radius for coordinate in xyz))
        else:
            points.append(xyz)

    if style.show_links:
        radius = (
            _nonnegative(layout.link_radius, "layout.link_radius")
            * _nonnegative(style.link_linewidth, "style.link_linewidth") / 3.0
        )
        if scene.dh_segments:
            for segment in scene.dh_segments:
                start = _vector3(segment.start, "DH segment start")
                end = _vector3(segment.end, "DH segment end")
                if start != end:  # Do not draw zero-length cylinders.
                    add_point(start, radius=radius, description="DH segment start")
                    add_point(end, radius=radius, description="DH segment end")
        else:
            for index in range(len(scene.joints) - 1):
                start = _vector3(scene.joints[index], "link start")
                end = _vector3(scene.joints[index + 1], "link end")
                if start != end:
                    add_point(start, radius=radius, description="link start")
                    add_point(end, radius=radius, description="link end")

    if style.show_joints:
        if style.joint_representation == "kinematic":
            if style.joint_appearance in ("schematic", "hybrid"):
                for symbol in build_joint_symbols(scene, layout=layout, style=style):
                    for path in symbol["paths"]:
                        for position in path:
                            add_point(position, description="joint symbol point")
            if style.joint_appearance in ("solid", "hybrid"):
                for solid in build_joint_solids(scene, layout=layout, style=style):
                    add_point(
                        solid["origin"],
                        radius=solid_bounding_radius(solid),
                        description="joint solid center",
                    )
            if style.show_base and scene.joints:
                add_point(
                    scene.joints[0],
                    radius=_nonnegative(layout.base_radius, "layout.base_radius"),
                    description="base position",
                )
        else:
            for index, position in enumerate(scene.joints):
                if index == 0 and not style.show_base:
                    continue
                radius_name = "layout.base_radius" if index == 0 else "layout.joint_radius"
                radius = _nonnegative(
                    layout.base_radius if index == 0 else layout.joint_radius,
                    radius_name,
                )
                add_point(position, radius=radius, description="joint position")

    if style.show_frames:
        length = _nonnegative(
            layout.frame_length if style.frame_scale is None else style.frame_scale,
            "frame_scale",
        )
        for frame in scene.frames:
            origin = _vector3(frame.position, "frame origin")
            add_point(origin)
            for direction in (frame.x, frame.y, frame.z):
                axis = _vector3(direction, "frame axis")
                add_point(
                    tuple(origin[i] + length * axis[i] for i in range(3)),
                    description="frame axis tip",
                )

    if style.show_trajectory and scene.frames:
        add_point(scene.frames[-1].position, description="end-effector trajectory")

    if not points:
        points.append((0.0, 0.0, 0.0))

    return SceneBounds(
        tuple(min(point[i] for point in points) for i in range(3)),
        tuple(max(point[i] for point in points) for i in range(3)),
    )


def resolve_framing(
    scenes: Sequence[SceneData],
    *,
    layout: VisualizationLayout | None = None,
    style: VisualizationStyle | None = None,
    margin: float = _DEFAULT_MARGIN,
    min_extent_factor: float = _DEFAULT_MIN_EXTENT_FACTOR,
) -> SceneFraming:
    """Resolve one stable view volume around all sampled scenes.

    Bounds cover only the supplied configurations, not continuous robot
    motion between sampled frames. Both half_extent (cube) and fit_radius
    (enclosing sphere) include a relative margin and a visual-only minimum
    proportional to the reference length, with no absolute physical floor.
    """
    if not scenes:
        raise ValueError("scenes must contain at least one scene.")
    if layout is None:
        layout = resolve_layout(scenes)
    if style is None:
        style = VisualizationStyle()

    margin = _nonnegative(margin, "margin")
    min_extent_factor = _nonnegative(min_extent_factor, "min_extent_factor")
    reference = _nonnegative(layout.reference_length, "layout.reference_length")
    if reference <= 0:
        raise ValueError("layout.reference_length must be positive.")

    bounds = union_bounds([
        compute_scene_bounds(scene, layout=layout, style=style)
        for scene in scenes
    ])
    half_sizes = tuple(value / 2.0 for value in bounds.size)
    min_half_extent = min_extent_factor * reference
    factor = 1.0 + margin

    half_extent = factor * max(*half_sizes, min_half_extent)
    fit_radius = factor * max(math.hypot(*half_sizes), min_half_extent)

    if not all(math.isfinite(x) and x > 0 for x in (half_extent, fit_radius)):
        raise ValueError("Framing sizes must be finite and positive.")

    return SceneFraming(
        bounds=bounds,
        center=bounds.center,
        half_extent=half_extent,
        fit_radius=fit_radius,
    )
