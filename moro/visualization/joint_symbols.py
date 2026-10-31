"""Backend-independent schematic symbols for classic-DH motion axes.

A symbol depicts the *allowed joint motion*, not actuator or CAD geometry.
Python creates the same world-space polylines for Matplotlib and Three.js.
"""

import math

import numpy as np

from .data import SceneData
from .layout import VisualizationLayout
from .style import VisualizationStyle


def _joint_plane(axis: np.ndarray, reference=None):
    """Return a right-handed orthonormal basis perpendicular to the axis."""
    z = np.asarray(axis, dtype=float)
    if reference is not None:
        x = np.asarray(reference, dtype=float)
        x = x - np.dot(x, z) * z
        length = np.linalg.norm(x)
        if np.isfinite(length) and length > 1e-12:
            x = x / length
            return x, np.cross(z, x)

    # Deterministic fallback for user-created SceneData without frames.
    basis = np.zeros(3)
    basis[int(np.argmin(np.abs(z)))] = 1.0
    x = basis - np.dot(basis, z) * z
    x /= np.linalg.norm(x)
    return x, np.cross(z, x)


def build_joint_symbols(
    scene: SceneData,
    *,
    layout: VisualizationLayout,
    style: VisualizationStyle,
) -> list[dict]:
    """Build one schematic polyline bundle per evaluated R/P joint.

    The return value is JSON-compatible. Every bundle has stable path counts
    and point counts across configurations of the same joint topology, which
    allows Three.js animations to update existing line buffers in place.

    Called only for the opt-in kinematic representation with visible joints.
    An explicit error avoids inventing joint types for legacy hand-built scenes.
    """
    if not style.show_joints or style.joint_representation != "kinematic":
        return []
    if not scene.joint_data:
        raise ValueError(
            "joint_representation='kinematic' requires SceneData.joint_data "
            "from evaluate_robot()."
        )

    radius = float(layout.joint_radius) * 2.8 * style.joint_scale
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Kinematic joint symbol radius must be finite and positive.")

    symbols = []

    for joint in scene.joint_data:
        origin = np.asarray(joint.origin, dtype=float)
        axis = np.asarray(joint.axis, dtype=float)
        reference = (
            scene.frames[joint.index - 1].x
            if len(scene.frames) >= joint.index
            else None
        )
        x, y = _joint_plane(axis, reference)

        def world(axial=0.0, dx=0.0, dy=0.0):
            return origin + axial * axis + dx * x + dy * y

        if joint.kind == "r":
            # 315-degree positive-rotation arc with a distinct arrowhead.
            angles = np.linspace(-math.pi / 4, 3 * math.pi / 2, 29)
            arc = [world(dx=radius * math.cos(t),
                         dy=radius * math.sin(t)) for t in angles]
            t = float(angles[-1])
            tip = arc[-1]
            tangent = -math.sin(t) * x + math.cos(t) * y
            inward = -math.cos(t) * x - math.sin(t) * y
            head = [
                tip - 0.33 * radius * tangent + 0.18 * radius * inward,
                tip,
                tip - 0.33 * radius * tangent - 0.18 * radius * inward,
            ]
            paths = [arc, head]
        elif joint.kind == "p":
            # Symbolic guide, centered slider and positive axial direction.
            half_length = radius * 1.3
            guide_width = radius * 0.52
            slider_half_length = radius * 0.38
            slider_half_width = radius * 0.72
            paths = [
                [world(-half_length, guide_width),
                 world(half_length, guide_width)],
                [world(-half_length, -guide_width),
                 world(half_length, -guide_width)],
                [
                    world(-slider_half_length, slider_half_width),
                    world(slider_half_length, slider_half_width),
                    world(slider_half_length, -slider_half_width),
                    world(-slider_half_length, -slider_half_width),
                    world(-slider_half_length, slider_half_width),
                ],
                [world(-half_length * 0.66),
                 world(half_length * 0.66)],
                [
                    world(half_length * 0.40, dx=radius * 0.24),
                    world(half_length * 0.66),
                    world(half_length * 0.40, dx=-radius * 0.24),
                ],
            ]
        else:
            raise ValueError("Unsupported joint type in joint_data.")

        symbols.append({
            "index": joint.index,
            "kind": joint.kind,
            "paths": [
                np.asarray(path, dtype=float).tolist() for path in paths
            ],
        })

    return symbols
