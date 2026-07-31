"""Visualization tools for serial robots."""

from .data import FrameData, SceneData
from .evaluation import evaluate_robot
from .matplotlib_backend import MatplotlibBackend
from .style import VisualizationStyle
from .threejs_backend import ThreeJSBackend
from .visualizer import RobotVisualizer

# Private helpers remain importable from ``moro.visualization`` temporarily so
# existing internal tests do not need to change during the package migration.
from .threejs_backend import (
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
    "VisualizationStyle",
]
