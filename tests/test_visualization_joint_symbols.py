"""V5-B2: shared schematic R/P symbols, rendering and animation contracts."""

import json
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

from moro.core import Robot
from moro.visualization import (
    FrameData, MatplotlibBackend, SceneData, ThreeJSBackend,
    VisualizationStyle, compute_scene_bounds, evaluate_robot, resolve_layout,
)
from moro.visualization.joint_symbols import build_joint_symbols
from moro.visualization.threejs_backend import _scene_to_payload


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _robot():
    q1, q2 = sp.symbols("q1 q2")
    robot = Robot((0.7, sp.pi / 2, 0.1, q1, "r"),
                  (0.4, 0, q2, 0, "p"))
    return robot, q1, q2


def _payload(html, variable):
    match = re.search(r"var " + variable + r" = (\{[^\n]+\});", html)
    assert match is not None
    return json.loads(match.group(1))


def _style(**kwargs):
    return VisualizationStyle(
        joint_representation="kinematic", show_links=False,
        show_frames=False, **kwargs
    )


def test_style_defaults_preserve_existing_spherical_markers():
    style = VisualizationStyle()
    assert style.joint_representation == "simple"
    assert style.joint_scale == 1.0
    assert style.show_joints
    assert style.show_base

def test_style_positional_compatibility_remains_intact():
    # The first five fields predate V5-B2; the sixth is link_color.
    style = VisualizationStyle(True, True, True, True, True, "#010203")
    assert style.link_color == "#010203"
    assert style.joint_representation == "simple"
    assert style.joint_scale == 1.0



@pytest.mark.parametrize("mode", ["kinematic2", "R", "p", "", None])
def test_invalid_joint_representation_rejected(mode):
    with pytest.raises(ValueError, match="joint_representation"):
        VisualizationStyle(joint_representation=mode)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "invalid"])
def test_invalid_joint_scale_rejected(scale):
    with pytest.raises(ValueError, match="joint_scale"):
        VisualizationStyle(joint_scale=scale)


def test_planar_revolute_symbol_is_perpendicular_to_joint_axis():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.8, 0, 0, q)), {q: 0.25})
    symbols = build_joint_symbols(
        scene, layout=resolve_layout([scene]), style=_style()
    )
    assert len(symbols) == 1
    assert symbols[0]["index"] == 1
    assert symbols[0]["kind"] == "r"
    assert len(symbols[0]["paths"]) == 2
    arc = np.asarray(symbols[0]["paths"][0])
    axis = scene.joint_data[0].axis
    origin = scene.joint_data[0].origin
    assert arc.shape == (29, 3)
    np.testing.assert_allclose((arc - origin) @ axis, 0, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(arc - origin, axis=1),
        resolve_layout([scene]).joint_radius * 2.8
    )


def test_spatial_rp_symbols_follow_motion_axes_not_end_frames():
    robot, q1, q2 = _robot()
    scene = evaluate_robot(robot, {q1: 0.4, q2: 0.3})
    symbols = build_joint_symbols(
        scene, layout=resolve_layout([scene]), style=_style()
    )
    assert [symbol["kind"] for symbol in symbols] == ["r", "p"]
    assert [symbol["index"] for symbol in symbols] == [1, 2]
    prismatic = symbols[1]
    assert [len(path) for path in prismatic["paths"]] == [2, 2, 5, 2, 3]
    first_rail = np.asarray(prismatic["paths"][0])
    movement = first_rail[1] - first_rail[0]
    direction = scene.joint_data[1].axis
    np.testing.assert_allclose(
        movement / np.linalg.norm(movement), direction, atol=1e-12
    )
    assert not np.allclose(scene.joint_data[1].origin, scene.frames[2].position)


def test_symbol_paths_scale_linearly_millimeters_vs_meters():
    q = sp.Symbol("q")
    paths = []
    for factor in (1, 1000):
        scene = evaluate_robot(
            Robot((0.6 * factor, sp.pi / 2, 0.2 * factor, q)), {q: 0.5}
        )
        paths.append(build_joint_symbols(
            scene, layout=resolve_layout([scene]), style=_style()
        ))
    for path_a, path_b in zip(paths[0][0]["paths"], paths[1][0]["paths"]):
        np.testing.assert_allclose(np.asarray(path_b), 1000 * np.asarray(path_a))


def test_joint_scale_controls_size_without_changing_anchor():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.5, 0, 0, q)), {q: 0})
    layout = resolve_layout([scene])
    a = build_joint_symbols(scene, layout=layout, style=_style(joint_scale=1))
    b = build_joint_symbols(scene, layout=layout, style=_style(joint_scale=2))
    origin = scene.joint_data[0].origin
    np.testing.assert_allclose(
        np.array(b[0]["paths"][0]) - origin,
        2 * (np.array(a[0]["paths"][0]) - origin),
    )


def test_no_joint_metadata_raises_only_when_kinematic_symbols_requested():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 1.0)
    layout = resolve_layout([scene])
    with pytest.raises(ValueError, match="joint_data"):
        build_joint_symbols(scene, layout=layout, style=_style())
    with pytest.raises(ValueError, match="joint_data"):
        MatplotlibBackend.render(scene, style=_style())
    with pytest.raises(ValueError, match="joint_data"):
        ThreeJSBackend.render(scene, style=_style())
    assert build_joint_symbols(
        scene, layout=layout, style=_style(show_joints=False)
    ) == []


def test_framing_covers_every_joint_symbol_vertex():
    robot, q1, q2 = _robot()
    scene = evaluate_robot(robot, {q1: -0.5, q2: -0.2})
    layout = resolve_layout([scene])
    style = _style(joint_scale=3.0, show_base=False)
    bounds = compute_scene_bounds(scene, layout=layout, style=style)
    symbols = build_joint_symbols(scene, layout=layout, style=style)
    for symbol in symbols:
        for path in symbol["paths"]:
            for point in path:
                assert all(bounds.minimum[i] <= point[i] <= bounds.maximum[i]
                           for i in range(3))


def test_matplotlib_kinematic_mode_draws_schematic_lines_and_base_only():
    robot, q1, q2 = _robot()
    scene = evaluate_robot(robot, {q1: 0.5, q2: 0.2})
    _, ax = MatplotlibBackend.render(scene, style=_style())
    # Revolute arc + arrowhead; prism guides, slider and arrow.
    assert len(ax.lines) == 7
    assert len(ax.collections) == 1  # Base marker only.
    expected = build_joint_symbols(
        scene, layout=resolve_layout([scene]), style=_style()
    )
    drawn = ax.lines[0].get_data_3d()
    np.testing.assert_allclose(np.asarray(drawn).T, expected[0]["paths"][0])


def test_matplotlib_hide_base_keeps_joint_1_symbol():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.5, 0, 0, q)), {q: 0})
    _, ax = MatplotlibBackend.render(scene, style=_style(show_base=False))
    assert len(ax.lines) == 2
    assert len(ax.collections) == 0


def test_matplotlib_legacy_mode_still_draws_frame_origin_spheres():
    robot, q1, q2 = _robot()
    scene = evaluate_robot(robot, {q1: 0.4, q2: 0.3})
    _, ax = MatplotlibBackend.render(
        scene, style=VisualizationStyle(show_links=False, show_frames=False)
    )
    assert len(ax.lines) == 0
    assert len(ax.collections) == 3


def test_matplotlib_kinematic_animation_has_stable_paths():
    robot, q1, q2 = _robot()
    scenes = [
        evaluate_robot(robot, {q1: angle, q2: travel})
        for angle, travel in ((0, 0), (0.4, 0.3), (-0.5, -0.2))
    ]
    style = _style(show_base=False)
    animation = MatplotlibBackend.animate(scenes, style=style)
    try:
        animation._init_func()
        for i, scene in enumerate(scenes):
            animation._func(i)
            paths = [path for symbol in build_joint_symbols(
                scene, layout=resolve_layout(scenes), style=style
            ) for path in symbol["paths"]]
            ax = animation._fig.axes[0]
            assert len(ax.lines) == len(paths)
            for line, path in zip(ax.lines, paths):
                np.testing.assert_allclose(np.asarray(line.get_data_3d()).T, path)
    finally:
        animation._draw_was_started = True


def test_threejs_opt_in_payload_matches_matplotlib_geometry():
    robot, q1, q2 = _robot()
    scene = evaluate_robot(robot, {q1: 0.35, q2: 0.2})
    style = _style()
    html = ThreeJSBackend.render(scene, style=style).data
    payload = _payload(html, "data")
    assert payload["style"]["joint_representation"] == "kinematic"
    assert payload["style"]["joint_scale"] == 1.0
    assert payload["joint_symbols"] == build_joint_symbols(
        scene, layout=resolve_layout([scene]), style=style
    )
    assert "createJointSymbolLines" in html
    assert "THREE.LineBasicMaterial" in html


def test_threejs_animation_paths_preserve_topology():
    robot, q1, q2 = _robot()
    scenes = [
        evaluate_robot(robot, {q1: angle, q2: travel})
        for angle, travel in ((0.0, 0.0), (0.6, 0.4), (-0.5, -0.1))
    ]
    html = ThreeJSBackend.animate(scenes, style=_style()).data
    payload = _payload(html, "animationData")
    assert all(len(frame["joint_symbols"]) == 2 for frame in payload["frames"])
    first = payload["frames"][0]["joint_symbols"]
    assert all(
        [(s["index"], s["kind"], list(map(len, s["paths"])))
         for s in frame["joint_symbols"]]
        == [(s["index"], s["kind"], list(map(len, s["paths"])))
            for s in first]
        for frame in payload["frames"]
    )
    assert "updateJointSymbolLines" in html
    assert "attribute.needsUpdate = true;" in html


def test_simple_threejs_payload_does_not_contain_symbol_geometry():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.5, 0, 0, q)), {q: 0})
    raw = _scene_to_payload(scene)
    assert "joint_symbols" not in raw
    payload = _payload(ThreeJSBackend.render(scene).data, "data")
    assert "joint_symbols" not in payload
    assert payload["style"]["joint_representation"] == "simple"


def test_hidden_joints_never_require_metadata_or_draw_symbols():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 1.0)
    style = _style(show_joints=False)
    assert "joint_symbols" not in _payload(
        ThreeJSBackend.render(scene, style=style).data, "data"
    )
    _, ax = MatplotlibBackend.render(scene, style=style)
    assert len(ax.collections) == 0
    assert len(ax.lines) == 0
