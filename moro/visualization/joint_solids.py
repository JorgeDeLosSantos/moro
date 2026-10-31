"""World-space schematic solids for an experimental Three.js joint appearance.

Dimensions and axes are computed in Python; Three.js only builds primitives
and applies transforms. Solids express kinematic joint kind, not CAD shapes.
"""

import math

import numpy as np

from .data import SceneData
from .layout import VisualizationLayout
from .joint_symbols import _joint_plane
from .style import VisualizationStyle


def build_joint_solids(
    scene: SceneData,
    *,
    layout: VisualizationLayout,
    style: VisualizationStyle,
) -> list[dict]:
    """Build proportional R cylinders and P rectangular prisms.

    Every primitive is centered at the previous DH frame origin and oriented
    along z_(i-1). The returned layout is JSON-compatible and has stable
    topology for animations. Solid dimensions never depend on the current
    prismatic extension.
    """
    if (
        not style.show_joints
        or style.joint_representation != "kinematic"
        or style.joint_appearance == "schematic"
    ):
        return []
    if not scene.joint_data:
        raise ValueError(
            "joint_representation='kinematic' requires SceneData.joint_data "
            "from evaluate_robot()."
        )

    # Reduce the prototype solid dimensions by 20%, without changing the
    # schematic stroke geometry or the meaning of joint_scale=1.0.
    radius = float(layout.joint_radius) * 2.8 * style.joint_scale * 0.8
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Kinematic joint solid dimensions must be finite and positive.")

    solids: list[dict] = []
    for joint in scene.joint_data:
        # Keep the prism faces aligned with the same transverse DH frame X
        # direction used for the schematic slider guides. The axis alone
        # cannot determine a cuboid's roll orientation.
        reference = (
            scene.frames[joint.index - 1].x
            if len(scene.frames) >= joint.index
            else None
        )
        transverse, _ = _joint_plane(joint.axis, reference)
        if joint.kind == "r":
            solid = {
                "index": joint.index,
                "kind": "r",
                "origin": np.asarray(joint.origin, dtype=float).tolist(),
                "axis": np.asarray(joint.axis, dtype=float).tolist(),
                "transverse": transverse.tolist(),
                "radius": radius * 0.72,
                "length": radius * 2.0,
            }
        elif joint.kind == "p":
            solid = {
                "index": joint.index,
                "kind": "p",
                "origin": np.asarray(joint.origin, dtype=float).tolist(),
                "axis": np.asarray(joint.axis, dtype=float).tolist(),
                "transverse": transverse.tolist(),
                "width": radius * 1.35,
                "depth": radius * 1.35,
                "length": radius * 2.6,
            }
        else:
            raise ValueError("Unsupported joint type in joint_data.")
        solids.append(solid)
    return solids


def solid_bounding_radius(solid: dict) -> float:
    """Conservative radius of a sphere enclosing an oriented solid."""
    half = solid["length"] / 2
    if solid["kind"] == "r":
        return math.hypot(half, solid["radius"])
    if solid["kind"] == "p":
        return math.sqrt(
            half * half +
            (solid["width"] / 2) ** 2 +
            (solid["depth"] / 2) ** 2
        )
    raise ValueError("Unsupported joint solid kind.")
