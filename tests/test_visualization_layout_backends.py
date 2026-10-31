"""V3-B regression tests for Matplotlib and Three.js layout adapters."""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    MatplotlibBackend, SceneData, FrameData, ThreeJSBackend,
    VisualizationStyle, evaluate_robot, resolve_layout,
)
from moro.visualization.threejs_backend import (
    _layout_to_payload, _scene_to_payload,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def payload_from_html(html, variable):
    match = re.search(r"var " + variable + r" = (\{[^\n]+\});", html)
    assert match is not None
    return json.loads(match.group(1))


def test_threejs_static_render_serializes_computed_world_layout():
    q = symbols("q")
    scene = evaluate_robot(Robot((0.7, 0, 0, q)), {q: 0.0})
    data = payload_from_html(ThreeJSBackend.render(scene).data, "data")
    assert data["layout"]["reference_length"] == pytest.approx(0.7)
    assert data["layout"]["frame_length"] == pytest.approx(0.7 / 6)
    assert data["layout"]["joint_radius"] == pytest.approx(0.014)
    assert data["layout"]["base_radius"] == pytest.approx(0.021)
    assert data["layout"]["link_radius"] == pytest.approx(0.0105)
    assert data["style"]["joint_size"] is None
    assert data["style"]["frame_scale"] is None


def test_threejs_animation_serializes_one_global_layout():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: x}) for x in (0.0, 0.1, 0.4, 0.2)]
    payload = payload_from_html(ThreeJSBackend.animate(scenes).data, "animationData")
    assert len(payload["frames"]) == 4
    assert payload["layout"]["reference_length"] == pytest.approx(0.9)
    assert payload["layout"]["joint_radius"] == pytest.approx(0.018)
    assert all("layout" not in frame for frame in payload["frames"])


def test_threejs_layout_is_proportional_between_meters_and_millimeters():
    q = symbols("q")
    layouts = []
    for scale in (0.7, 700.0):
        scene = evaluate_robot(Robot((scale, 0, 0, q)), {q: 0.0})
        layouts.append(
            payload_from_html(ThreeJSBackend.render(scene).data, "data")["layout"]
        )
    for key in ("reference_length", "frame_length", "joint_radius",
                "base_radius", "link_radius"):
        assert layouts[1][key] == pytest.approx(layouts[0][key] * 1000)


def test_threejs_explicit_style_preserved_and_python_layout_used():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 1.5)
    layout = resolve_layout([scene], reference_length=2.5)
    style = VisualizationStyle(frame_scale=1.2, joint_size=0.7,
                               base_size=0.9, link_linewidth=5)
    result = ThreeJSBackend.render(scene, style=style, layout=layout).data
    payload = payload_from_html(result, "data")
    assert payload["layout"]["reference_length"] == 2.5
    assert payload["style"]["frame_scale"] == 1.2
    assert payload["style"]["joint_size"] == 0.7
    assert payload["style"]["base_size"] == 0.9
    assert payload["style"]["link_linewidth"] == 5
    assert "style.joint_size != null" in result
    assert "autoSize('joint_radius'" in result


def test_threejs_legacy_scene_payload_contract_unchanged():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 2)
    assert set(_scene_to_payload(scene)) == {"joints", "frames", "dimension"}
    assert set(_layout_to_payload(resolve_layout([scene]))) == {
        "reference_length", "frame_length", "joint_radius",
        "base_radius", "link_radius",
    }


def test_matplotlib_default_markers_are_unit_invariant():
    q = symbols("q")
    sizes = []
    for scale in (0.1, 100.0):
        scene = evaluate_robot(Robot((scale, 0, 0, q)), {q: 0.0})
        fig, ax = MatplotlibBackend.render(
            scene, style=VisualizationStyle(show_frames=False)
        )
        assert len(ax.collections) == 2
        sizes.append([float(collection.get_sizes()[0]) for collection in ax.collections])
        plt.close(fig)
    np.testing.assert_allclose(sizes[0], sizes[1], rtol=1e-10)


def test_matplotlib_respects_explicit_marker_sizes():
    T1 = np.eye(4)
    T1[0, 3] = 1
    scene = SceneData(
        [np.zeros(3), np.array([1.0, 0, 0])],
        [FrameData(np.eye(4)), FrameData(T1)], 2.0,
    )
    style = VisualizationStyle(
        frame_scale=0.4, joint_size=0.3, base_size=0.6,
        show_links=False, show_frames=False,
    )
    fig, ax = MatplotlibBackend.render(scene, style=style)
    assert [float(s.get_sizes()[0]) for s in ax.collections] == pytest.approx(
        [60.0, 30.0]
    )
    plt.close(fig)


def test_matplotlib_default_frame_length_comes_from_layout():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 2.0)
    layout = resolve_layout([scene], reference_length=3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    actual_lengths = []
    original_quiver = ax.quiver

    def record_quiver(*args, **kwargs):
        actual_lengths.append(kwargs["length"])
        return original_quiver(*args, **kwargs)

    ax.quiver = record_quiver
    MatplotlibBackend.render(scene, ax=ax, layout=layout,
                             style=VisualizationStyle(show_joints=False,
                                                      show_links=False))
    assert actual_lengths == pytest.approx([0.5] * 3)
    plt.close(fig)


def test_matplotlib_explicit_frame_length_overrides_layout():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 2)
    layout = resolve_layout([scene], reference_length=3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    recorded = []
    original_quiver = ax.quiver

    def record_quiver(*args, **kwargs):
        recorded.append(kwargs["length"])
        return original_quiver(*args, **kwargs)

    ax.quiver = record_quiver
    MatplotlibBackend.render(scene, ax=ax, layout=layout,
                             style=VisualizationStyle(frame_scale=0.25,
                                                      show_joints=False,
                                                      show_links=False))
    assert recorded == pytest.approx([0.25] * 3)
    plt.close(fig)


def test_matplotlib_animation_reuses_global_layout():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: d}) for d in (0, 0.1, 0.4, 0.2)]
    animation = MatplotlibBackend.animate(
        scenes, style=VisualizationStyle(show_frames=False),
    )
    animation._init_func()
    animation._func(0)
    sizes_first = [s.get_sizes()[0] for s in animation._fig.axes[0].collections]
    animation._func(2)
    sizes_later = [s.get_sizes()[0] for s in animation._fig.axes[0].collections]
    np.testing.assert_allclose(sizes_first, sizes_later)
    animation._draw_was_started = True
