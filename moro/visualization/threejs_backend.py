"""Three.js visualization backend and HTML serialization helpers."""

import json
import re
import uuid

from importlib.resources import files
from typing import Any

import numpy as np

from .data import SceneData
from .framing import SceneFraming, resolve_framing
from .layout import VisualizationLayout, resolve_layout
from .joint_symbols import build_joint_symbols
from .joint_solids import build_joint_solids
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
    """Serialize a scene, including evaluated DH geometry when available.

    Legacy scenes keep their original payload shape, and explicit zero-length
    DH segments remain present to preserve topological identity in animations.
    """
    payload = {
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
    if scene_data.dh_segments:
        payload["dh_segments"] = [
            {
                "row_index": int(segment.row_index),
                "kind": segment.kind,
                "start": np.asarray(segment.start, dtype=float).tolist(),
                "end": np.asarray(segment.end, dtype=float).tolist(),
            }
            for segment in scene_data.dh_segments
        ]
    return payload


def _framing_to_payload(framing: SceneFraming) -> dict:
    """Serialize camera-independent world-space bounds and framing."""
    return {
        "minimum": list(framing.bounds.minimum),
        "maximum": list(framing.bounds.maximum),
        "center": list(framing.center),
        "half_extent": float(framing.half_extent),
        "fit_radius": float(framing.fit_radius),
    }


def _layout_to_payload(layout: VisualizationLayout) -> dict:
    """Serialize pre-resolved geometric sizes in robot linear units."""
    return {
        "reference_length": float(layout.reference_length),
        "frame_length": float(layout.frame_length),
        "joint_radius": float(layout.joint_radius),
        "base_radius": float(layout.base_radius),
        "link_radius": float(layout.link_radius),
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
        "joint_representation": style.joint_representation,
        "joint_scale": style.joint_scale,
        "joint_appearance": style.joint_appearance,
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
        layout: VisualizationLayout | None = None,
        framing: SceneFraming | None = None,
    ):
        """Return an interactive Three.js view as ``IPython.display.HTML``."""
        from IPython.display import HTML

        if style is None:
            style = VisualizationStyle()

        if layout is None:
            layout = resolve_layout([scene_data])

        if framing is None:
            framing = resolve_framing([scene_data], layout=layout, style=style)

        payload = _scene_to_payload(scene_data)
        if style.show_joints and style.joint_representation == "kinematic":
            if style.joint_appearance in ("schematic", "hybrid"):
                payload["joint_symbols"] = build_joint_symbols(
                    scene_data, layout=layout, style=style
                )
            if style.joint_appearance in ("solid", "hybrid"):
                payload["joint_solids"] = build_joint_solids(
                    scene_data, layout=layout, style=style
                )
        payload["style"] = _style_to_payload(style)
        payload["layout"] = _layout_to_payload(layout)
        payload["framing"] = _framing_to_payload(framing)

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
        layout: VisualizationLayout | None = None,
        framing: SceneFraming | None = None,
    ):
        """Return an interactive Three.js animation as ``IPython.display.HTML``."""
        from IPython.display import HTML

        if not scene_data_list:
            raise ValueError(
                "scene_data_list must contain at least one scene."
            )

        if style is None:
            style = VisualizationStyle()

        if layout is None:
            layout = resolve_layout(scene_data_list)

        if framing is None:
            framing = resolve_framing(
                scene_data_list, layout=layout, style=style
            )

        frames_payload = _scenes_to_payload(scene_data_list)
        if style.show_joints and style.joint_representation == "kinematic":
            for scene, payload in zip(scene_data_list, frames_payload):
                if style.joint_appearance in ("schematic", "hybrid"):
                    payload["joint_symbols"] = build_joint_symbols(
                        scene, layout=layout, style=style
                    )
                if style.joint_appearance in ("solid", "hybrid"):
                    payload["joint_solids"] = build_joint_solids(
                        scene, layout=layout, style=style
                    )
            if style.joint_appearance in ("schematic", "hybrid"):
                reference = frames_payload[0]["joint_symbols"]
                for frame_index, payload in enumerate(frames_payload[1:], start=1):
                    actual = payload["joint_symbols"]
                    if len(actual) != len(reference) or any(
                        a["index"] != b["index"] or a["kind"] != b["kind"] or
                        tuple(map(len, a["paths"])) != tuple(map(len, b["paths"]))
                        for a, b in zip(reference, actual)
                    ):
                        raise ValueError(
                            "Joint symbol topology differs across animation frames "
                            f"(frame {frame_index})."
                        )
            if style.joint_appearance in ("solid", "hybrid"):
                reference = frames_payload[0]["joint_solids"]
                for frame_index, payload in enumerate(frames_payload[1:], start=1):
                    actual = payload["joint_solids"]
                    if len(actual) != len(reference) or any(
                        a["index"] != b["index"] or a["kind"] != b["kind"]
                        for a, b in zip(reference, actual)
                    ):
                        raise ValueError(
                            "Joint solid topology differs across animation frames "
                            f"(frame {frame_index})."
                        )

        animation_payload = {
            "frames": frames_payload,
            "style": _style_to_payload(style),
            "layout": _layout_to_payload(layout),
            "framing": _framing_to_payload(framing),
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
