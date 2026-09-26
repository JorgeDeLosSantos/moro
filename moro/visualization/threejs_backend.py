"""Three.js visualization backend and HTML serialization helpers."""

import json
import re
import uuid

from importlib.resources import files
from typing import Any

import numpy as np

from .data import SceneData
from .style import VisualizationStyle


_PLACEHOLDER_PATTERN = re.compile(r"__[A-Z][A-Z0-9_]*__")


def _load_template_resource(resource_name: str) -> str:
    """Load one resource from ``moro.visualization.templates``."""
    return (
        files("moro.visualization.templates")
        .joinpath(resource_name)
        .read_text(encoding="utf-8")
    )


def _replace_placeholders(
    template: str,
    replacements: dict[str, Any],
) -> str:
    """Replace ``__PLACEHOLDER__`` tokens in a template string."""
    html = template

    for name, value in replacements.items():
        placeholder = f"__{name.upper()}__"

        if placeholder not in html:
            raise ValueError(
                f"Placeholder {placeholder!r} was not found in template."
            )

        html = html.replace(placeholder, str(value))

    unresolved = sorted(set(_PLACEHOLDER_PATTERN.findall(html)))
    if unresolved:
        raise ValueError(
            f"Unresolved placeholders in template: {unresolved}"
        )

    return html


def _render_html_template(
    template_name: str,
    replacements: dict[str, Any],
) -> str:
    """Load and render a template from ``moro.visualization.templates``."""
    template = _load_template_resource(template_name)
    replacement_values = dict(replacements)

    if "__COMMON_SCRIPT__" in template and "common_script" not in replacement_values:
        replacement_values["common_script"] = _load_template_resource(
            "threejs_common.js"
        )

    return _replace_placeholders(template, replacement_values)


def _scene_to_payload(scene_data: SceneData) -> dict:
    """Convert one scene into a JSON-serializable dictionary."""
    return {
        "joints": [
            np.asarray(joint, dtype=float).tolist()
            for joint in scene_data.joints
        ],
        "frames": [
            {
                "position": frame.position.tolist(),
                "x": frame.x.tolist(),
                "y": frame.y.tolist(),
                "z": frame.z.tolist(),
            }
            for frame in scene_data.frames
        ],
        "dimension": float(scene_data.dimension),
    }


def _scenes_to_payload(scene_data_list: list[SceneData]) -> list[dict]:
    """Convert multiple scenes into JSON-serializable dictionaries."""
    return [_scene_to_payload(scene) for scene in scene_data_list]


def _extract_end_effector_trajectory(
    scene_data_list: list[SceneData],
) -> list[list[float]]:
    """Extract end-effector Cartesian positions from an animation sequence."""
    trajectory: list[list[float]] = []

    for index, scene in enumerate(scene_data_list):
        if not scene.frames:
            raise ValueError(
                "Each scene must contain at least one frame to extract "
                "end-effector trajectory "
                f"(empty frames at index {index})."
            )

        trajectory.append(
            scene.frames[-1].position.astype(float).tolist()
        )

    return trajectory


def _style_to_payload(style: VisualizationStyle) -> dict:
    """Convert a visualization style into a JSON-serializable dictionary."""
    return {
        "show_frames": style.show_frames,
        "show_links": style.show_links,
        "show_joints": style.show_joints,
        "show_base": style.show_base,
        "show_grid": style.show_grid,
        "link_color": style.link_color,
        "joint_color": style.joint_color,
        "base_color": style.base_color,
        "frame_scale": style.frame_scale,
        "joint_size": style.joint_size,
        "base_size": style.base_size,
        "link_linewidth": style.link_linewidth,
        "show_trajectory": style.show_trajectory,
        "trajectory_color": style.trajectory_color,
        "trajectory_linewidth": style.trajectory_linewidth,
        "trajectory_mode": style.trajectory_mode,
    }


class ThreeJSBackend:
    """Render robot kinematic diagrams using Three.js."""

    @staticmethod
    def render(
        scene_data: SceneData,
        width: int = 800,
        height: int = 600,
        style: VisualizationStyle | None = None,
    ):
        """Return an interactive Three.js view as ``IPython.display.HTML``."""
        from IPython.display import HTML

        if style is None:
            style = VisualizationStyle()

        payload = _scene_to_payload(scene_data)
        payload["style"] = _style_to_payload(style)

        html = _render_html_template(
            "threejs_viewer.html",
            {
                "unique_id": uuid.uuid4().hex[:8],
                "width": width,
                "height": height,
                "robot_data": json.dumps(payload),
            },
        )
        return HTML(html)

    @staticmethod
    def animate(
        scene_data_list: list[SceneData],
        width: int = 800,
        height: int = 600,
        style: VisualizationStyle | None = None,
    ):
        """Return an interactive Three.js animation as ``IPython.display.HTML``."""
        from IPython.display import HTML

        if not scene_data_list:
            raise ValueError(
                "scene_data_list must contain at least one scene."
            )

        if style is None:
            style = VisualizationStyle()

        frames_payload = _scenes_to_payload(scene_data_list)
        animation_payload = {
            "frames": frames_payload,
            "style": _style_to_payload(style),
            "trajectory": _extract_end_effector_trajectory(scene_data_list),
        }

        html = _render_html_template(
            "threejs_animation.html",
            {
                "unique_id": uuid.uuid4().hex[:8],
                "width": width,
                "height": height,
                "animation_data": json.dumps(animation_payload),
                "last_frame": len(frames_payload) - 1,
            },
        )
        return HTML(html)
