"""V5-C2 public default-backend, auto style and animation interval contracts."""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    FrameData,
    MatplotlibBackend,
    RobotVisualizer,
    SceneData,
    ThreeJSBackend,
    VisualizationStyle,
    evaluate_robot,
)
from moro.visualization.style import resolve_joint_style


def _payload(html, variable):
    match = re.search(r"var " + variable + r" = (\{[^\n]+\});", html)
    assert match, "Serialized payload is missing."
    return json.loads(match.group(1))


def _robot():
    q1, d2 = symbols("q1 d2")
    robot = Robot(
        (0.7, 0, 0.2, q1, "r"),
        (0.4, 0, d2, 0, "p"),
    )
    return robot, q1, d2


def _legacy():
    identity = np.eye(4)
    return SceneData(
        joints=[np.zeros(3), np.array([0.5, 0, 0])],
        frames=[FrameData(identity), FrameData(identity)],
        dimension=1.0,
    )


def test_plot_defaults_to_threejs_with_solid_rp_metadata():
    robot, q1, d2 = _robot()
    out = RobotVisualizer(robot).plot({q1: 0.5, d2: 0.25})
    assert isinstance(out, HTML)
    data = _payload(out.data, "data")
    assert data["style"]["joint_representation"] == "kinematic"
    assert data["style"]["joint_appearance"] == "solid"
    assert [solid["kind"] for solid in data["joint_solids"]] == ["r", "p"]
    assert "joint_symbols" not in data


def test_animate_defaults_to_threejs_and_preserves_solid_topology():
    robot, q1, d2 = _robot()
    out = RobotVisualizer(robot).animate([
        {q1: 0, d2: 0.1}, {q1: 0.5, d2: 0.45},
    ])
    assert isinstance(out, HTML)
    data = _payload(out.data, "animationData")
    assert data["style"]["joint_representation"] == "kinematic"
    assert data["interval"] == 100
    assert len(data["frames"]) == 2
    assert all(
        [s["kind"] for s in frame["joint_solids"]] == ["r", "p"]
        for frame in data["frames"]
    )


def test_custom_style_uses_auto_without_being_mutated():
    robot, q1, d2 = _robot()
    style = VisualizationStyle(show_frames=False, joint_scale=1.3)
    data = _payload(
        RobotVisualizer(robot).plot(
            {q1: 0.3, d2: 0.1}, style=style
        ).data,
        "data",
    )
    assert data["style"]["joint_representation"] == "kinematic"
    assert data["style"]["show_frames"] is False
    assert data["style"]["joint_scale"] == 1.3
    assert style.joint_representation == "auto"


@pytest.mark.parametrize("mode,appearance,solid,symbols", [
    ("auto", "schematic", False, True),
    ("simple", "solid", False, False),
    ("kinematic", "solid", True, False),
    ("kinematic", "schematic", False, True),
])
def test_threejs_explicit_joint_representation_overrides(mode,appearance,solid,symbols):
    robot, q1, d2 = _robot()
    style = VisualizationStyle(
        joint_representation=mode, joint_appearance=appearance
    )
    payload = _payload(
        RobotVisualizer(robot).plot(
            {q1: 0.2, d2: 0.4}, style=style
        ).data, "data"
    )
    assert ("joint_solids" in payload) is solid
    assert ("joint_symbols" in payload) is symbols


def test_legacy_scenes_auto_fall_back_to_spheres():
    scene = _legacy()
    default = _payload(ThreeJSBackend.render(scene).data, "data")
    assert default["style"]["joint_representation"] == "simple"
    assert "joint_solids" not in default
    anim = _payload(ThreeJSBackend.animate([scene, scene]).data, "animationData")
    assert anim["style"]["joint_representation"] == "simple"
    assert all("joint_solids" not in f for f in anim["frames"])


def test_mixed_scene_animation_auto_falls_back_to_simple():
    robot, q1, d2 = _robot()
    modern = evaluate_robot(robot, {q1: 0.3, d2: 0.5})
    resolved = resolve_joint_style(
        VisualizationStyle(), backend="threejs",
        scenes=[modern, _legacy()],
    )
    assert resolved.joint_representation == "simple"


def test_explicit_kinematic_remains_strict_on_legacy_scenes():
    scene = _legacy()
    with pytest.raises(ValueError, match="joint_data"):
        ThreeJSBackend.render(
            scene, style=VisualizationStyle(joint_representation="kinematic")
        )


def test_matplotlib_explicit_still_returns_axes_and_animation():
    robot, q1, d2 = _robot()
    viz = RobotVisualizer(robot)
    values = {q1: 0.1, d2: 0.4}
    fig, ax = viz.plot(values, backend="matplotlib")
    assert ax.figure is fig
    plt.close(fig)
    anim = viz.animate(
        [values, {q1: 0.5, d2: 0.3}], backend="matplotlib", interval=55
    )
    assert isinstance(anim, FuncAnimation)
    assert anim.event_source.interval == 55
    anim._draw_was_started = True


def test_matplotlib_resolves_auto_to_simple_even_with_joint_metadata():
    robot, q1, d2 = _robot()
    scene = evaluate_robot(robot, {q1: 0.3, d2: 0.2})
    style = VisualizationStyle()
    effective = resolve_joint_style(style, backend="matplotlib", scenes=[scene])
    assert effective.joint_representation == "simple"
    assert style.joint_representation == "auto"
    fig, ax = MatplotlibBackend.render(scene, style=style)
    assert fig is ax.figure
    plt.close(fig)


def test_explicit_matplotlib_kinematic_schematic_still_supported():
    robot, q1, d2 = _robot()
    style = VisualizationStyle(
        joint_representation="kinematic", joint_appearance="schematic"
    )
    fig, ax = RobotVisualizer(robot).plot(
        {q1: 0.5, d2: 0.1}, backend="matplotlib", style=style
    )
    assert len(ax.lines) > 0
    plt.close(fig)


@pytest.mark.parametrize("interval", [1, 33.3, 100, 500])
def test_threejs_interval_is_serialized_to_animation_and_js(interval):
    robot, q1, d2 = _robot()
    html = RobotVisualizer(robot).animate(
        [{q1: 0, d2: 0}, {q1: 0.1, d2: 0.2}], interval=interval
    ).data
    payload = _payload(html, "animationData")
    assert payload["interval"] == pytest.approx(interval)
    assert "animationData.interval === undefined ? 100 : animationData.interval" in html


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"),
                                      float("-inf"), True, None, "fast"])
def test_threejs_invalid_interval_rejected(interval):
    robot, q1, d2 = _robot()
    with pytest.raises(ValueError, match="interval"):
        RobotVisualizer(robot).animate([{q1: 0, d2: 0}], interval=interval)


def test_unknown_backends_and_invalid_mode_still_rejected():
    robot, q1, d2 = _robot()
    with pytest.raises(ValueError, match="Unknown backend"):
        RobotVisualizer(robot).plot({q1: 0, d2: 0}, backend="bad")
    with pytest.raises(ValueError, match="joint_representation"):
        VisualizationStyle(joint_representation="hybrid")
