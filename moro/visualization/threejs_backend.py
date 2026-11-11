"""Three.js visualization backend and HTML serialization helpers."""

import json
import math
import re
import uuid
from numbers import Real

from importlib.resources import files
from typing import Any

import numpy as np

from .data import SceneData
from .framing import SceneFraming, resolve_framing
from .layout import VisualizationLayout, resolve_layout
from .playback import _normalize_playback_speed, _normalize_playback_times
from .joint_symbols import build_joint_symbols
from .joint_solids import build_joint_solids
from .style import VisualizationStyle, resolve_joint_style


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
        "revolute_color": style.revolute_color,
        "prismatic_color": style.prismatic_color,
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
        style = resolve_joint_style(
            style, backend="threejs", scenes=[scene_data]
        )

        if layout is None:
            layout = resolve_layout([scene_data])

        if framing is None:
            framing = resolve_framing([scene_data], layout=layout, style=style)

        payload = _scene_to_payload(scene_data)
        if style.show_joints and style.joint_representation == "kinematic":
            if style.joint_appearance == "schematic":
                payload["joint_symbols"] = build_joint_symbols(
                    scene_data, layout=layout, style=style
                )
            if style.joint_appearance == "solid":
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
        interval: float = 100,
        relative_times=None,
        playback_speed: float = 1.0,
    ):
        """Return a Three.js animation; optionally follow physical timestamps.

        relative_times is a private normalized-seconds input from the
        visualizer; without it, the historical fixed-interval mode is kept.
        """
        from IPython.display import HTML

        if not scene_data_list:
            raise ValueError(
                "scene_data_list must contain at least one scene."
            )
        if isinstance(interval, bool) or not isinstance(interval, Real):
            raise ValueError("interval must be a finite positive number of milliseconds.")
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be a finite positive number of milliseconds.")

        speed = _normalize_playback_speed(playback_speed)
        if relative_times is None:
            if speed != 1.0:
                raise ValueError(
                    "playback_speed requires physical-time timestamps."
                )
            times_payload = None
        else:
            times_payload = _normalize_playback_times(
                relative_times, len(scene_data_list)
            ).tolist()

        if style is None:
            style = VisualizationStyle()
        style = resolve_joint_style(
            style, backend="threejs", scenes=scene_data_list
        )

        if layout is None:
            layout = resolve_layout(scene_data_list)

        if framing is None:
            framing = resolve_framing(
                scene_data_list, layout=layout, style=style
            )

        frames_payload = _scenes_to_payload(scene_data_list)
        if style.show_joints and style.joint_representation == "kinematic":
            for scene, payload in zip(scene_data_list, frames_payload):
                if style.joint_appearance == "schematic":
                    payload["joint_symbols"] = build_joint_symbols(
                        scene, layout=layout, style=style
                    )
                if style.joint_appearance == "solid":
                    payload["joint_solids"] = build_joint_solids(
                        scene, layout=layout, style=style
                    )
            if style.joint_appearance == "schematic":
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
            if style.joint_appearance == "solid":
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
            "interval": interval,
        }
        if times_payload is not None:
            animation_payload["relative_times"] = times_payload
            animation_payload["playback_speed"] = speed

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


def _workspace_point_payload(
    points, *, projection, plane_level=None, allow_empty=False
):
    """Build viewer positions and framing without modifying source coordinates.

    A section is projected onto its *nominal* plane level, rather than
    the average of its sampled slab. Empty sections use a deterministic
    unit-scale reference view in that plane.
    """
    values = np.asarray(points)
    if np.iscomplexobj(values):
        raise ValueError("workspace points must contain real values.")
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("workspace points must contain finite real values.") from exc
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("workspace points must have shape (N, 3).")
    if not np.isfinite(values).all():
        raise ValueError("workspace points must contain finite real values.")
    if len(values) == 0 and not allow_empty:
        raise ValueError("workspace points must have at least one row.")
    normal = {"xy": 2, "xz": 1, "yz": 0}.get(projection)
    if normal is None and projection != "3d":
        raise ValueError("projection must be 'xy', 'xz', 'yz', or '3d'.")
    if plane_level is not None:
        if normal is None or not math.isfinite(float(plane_level)):
            raise ValueError("plane_level requires a planar projection and finite level.")
        plane_level = float(plane_level)

    display = values.copy()
    if len(display):
        if normal is not None:
            display[:, normal] = (
                plane_level if plane_level is not None
                else (display[:, normal].min() + display[:, normal].max()) / 2.0
            )
        minimum = display.min(axis=0)
        maximum = display.max(axis=0)
        center = minimum / 2.0 + maximum / 2.0
        span = maximum - minimum
        radius = max(
            0.6 * float(np.linalg.norm(span)),
            1e-6 * max(float(np.max(np.abs(center))), 1e-6),
        )
    else:
        # Empty slab sections are meaningful: show their named plane
        # and requested level, even when there are no selected samples.
        center = np.zeros(3, dtype=float)
        if normal is not None:
            center[normal] = plane_level if plane_level is not None else 0.0
        radius = 1.0
        minimum = center.copy()
        maximum = center.copy()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Could not compute a finite workspace viewing scale.")
    return {
        "positions": display.tolist(),
        "projection": projection,
        "framing": {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "center": center.tolist(),
            "fit_radius": radius,
        },
    }


def render_workspace_points(
    points, *, projection, marker_size, alpha, width, height,
    show_grid=False, plane_level=None, section_info=None,
):
    """Return an interactive Three.js workspace point cloud as IPython HTML."""
    from IPython.display import HTML

    for name, value in (("width", width), ("height", height)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(float(marker_size)) or float(marker_size) <= 0:
        raise ValueError("marker_size must be positive and finite.")
    if not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
        raise ValueError("alpha must be finite and satisfy 0 <= alpha <= 1.")
    if not isinstance(show_grid, (bool, np.bool_)):
        raise ValueError("show_grid must be Boolean.")

    payload = _workspace_point_payload(
        points,
        projection=projection,
        plane_level=plane_level,
        allow_empty=section_info is not None,
    )
    payload["marker_size"] = float(marker_size)
    payload["alpha"] = float(alpha)
    payload["show_grid"] = bool(show_grid)
    if section_info is not None:
        payload["section"] = dict(section_info)
    html = _render_html_template(
        "threejs_workspace.html",
        {
            "unique_id": uuid.uuid4().hex[:8],
            "width": int(width),
            "height": int(height),
            "workspace_data": json.dumps(payload),
        },
    )
    return HTML(html)
