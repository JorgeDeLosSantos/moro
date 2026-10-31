"""Visualization tools for serial robots."""

from .data import DHSegmentData, FrameData, SceneData
from .evaluation import evaluate_robot
from .framing import SceneBounds, SceneFraming, compute_scene_bounds, resolve_framing, union_bounds
from .layout import VisualizationLayout, resolve_layout
from .scaling import resolve_reference_length
from .matplotlib_backend import MatplotlibBackend
from .style import VisualizationStyle
from .threejs_backend import ThreeJSBackend
from .visualizer import RobotVisualizer
from .workspace import plot_workspace

# Private helpers remain importable from ``moro.visualization`` temporarily so
# existing internal tests do not need to change during the package migration.
from .threejs_backend import (
    _extract_end_effector_trajectory,
    _render_html_template,
    _replace_placeholders,
    _scene_to_payload,
    _scenes_to_payload,
    _style_to_payload,
)


__all__ = [
    "RobotVisualizer",
    "MatplotlibBackend",
    "ThreeJSBackend",
    "evaluate_robot",
    "SceneData",
    "FrameData",
    "DHSegmentData",
    "resolve_reference_length",
    "VisualizationLayout",
    "resolve_layout",
    "SceneBounds",
    "SceneFraming",
    "compute_scene_bounds",
    "union_bounds",
    "resolve_framing",
    "VisualizationStyle",
    "plot_workspace",
]
