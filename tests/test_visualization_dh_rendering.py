"""V5-A: draw evaluated DH segments in both static and animated views."""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp
from IPython.display import HTML

from moro.core import Robot
from moro.visualization import (
    DHSegmentData, FrameData, MatplotlibBackend, SceneData, ThreeJSBackend,
    VisualizationStyle, evaluate_robot,
)
from moro.visualization.threejs_backend import (
    _load_template_resource, _scene_to_payload, _scenes_to_payload,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _plot_lines(scene, **kwargs):
    _, ax = MatplotlibBackend.render(
        scene,
        style=VisualizationStyle(show_joints=False, show_frames=False),
        **kwargs,
    )
    return [
        np.array([line.get_data_3d()[0], line.get_data_3d()[1],
                  line.get_data_3d()[2]]).T
        for line in ax.lines
    ]


def _html_payload(html, variable):
    pattern = r"var " + variable + r" = (\{[^\n]+\});"
    result = re.search(pattern, html)
    assert result is not None
    return json.loads(result.group(1))


def _animated_scene(robot, variables):
    return [evaluate_robot(robot, values) for values in variables]


def test_matplotlib_planar_rr_uses_nonzero_a_segments_only():
    q1, q2 = sp.symbols("q1 q2")
    scene = evaluate_robot(
        Robot((1, 0, 0, q1), (0.5, 0, 0, q2)),
        {q1: 0.3, q2: -0.5},
    )
    assert len(scene.dh_segments) == 4
    lines = _plot_lines(scene)
    assert len(lines) == 2
    for drawn, segment in zip(lines, scene.dh_segments[1::2]):
        np.testing.assert_allclose(drawn, [segment.start, segment.end])


def test_matplotlib_spatial_dh_offset_and_length_not_diagonal():
    q = sp.Symbol("q")
    robot = Robot((2.0, sp.pi / 2, 3.0, q))
    scene = evaluate_robot(robot, {q: .4})
    assert len(scene.dh_segments) == 2
    lines = _plot_lines(scene)
    assert len(lines) == 2
    for line, seg in zip(lines, scene.dh_segments):
        np.testing.assert_allclose(line, [seg.start, seg.end])
    # The actual link contains the DH corner, not a diagonal between joints.
    assert not np.allclose(scene.dh_segments[0].end, scene.joints[1])


def test_matplotlib_negative_signed_dh_translations():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((-2, 0, -3, q)), {q: 0.1})
    lines = _plot_lines(scene)
    assert len(lines) == 2
    for line, seg in zip(lines, scene.dh_segments):
        np.testing.assert_allclose(line, [seg.start, seg.end])


def test_matplotlib_tiny_nonzero_dh_link_has_no_absolute_cutoff():
    q = sp.Symbol("q")
    for length in (1e-9, 1e-3, 1, 1000):
        scene = evaluate_robot(Robot((length, 0, 0, q)), {q: 0.0})
        lines = _plot_lines(scene)
        assert len(lines) == 1
        np.testing.assert_allclose(
            lines[0], [scene.dh_segments[1].start, scene.dh_segments[1].end]
        )


def test_matplotlib_zero_length_dh_geometry_keeps_markers_but_no_links():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0, 0, 0, q)), {q: 0.0})
    assert len(scene.dh_segments) == 2
    assert _plot_lines(scene) == []
    _, ax = MatplotlibBackend.render(
        scene, style=VisualizationStyle(show_frames=False)
    )
    assert len(ax.collections) == 2
    assert len(ax.lines) == 0


def test_matplotlib_legacy_joint_polyline_is_retained():
    T = np.eye(4)
    scene = SceneData(
        [np.array([0., 0., 0.]), np.array([1., 2., 3.])],
        [FrameData(T)], 3.0,
    )
    assert "dh_segments" not in _scene_to_payload(scene)
    lines = _plot_lines(scene)
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0], scene.joints)


def test_matplotlib_hide_links_still_hides_dh_segments():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, 0, 0.5, q)), {q: 0})
    _, ax = MatplotlibBackend.render(
        scene, style=VisualizationStyle(
            show_links=False, show_frames=False, show_joints=False
        ),
    )
    assert len(ax.lines) == 0


def test_matplotlib_animation_updates_dh_segments_without_diagonal():
    q = sp.Symbol("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = _animated_scene(robot, [{q: 0.0}, {q: 0.25}, {q: -0.1}])
    style = VisualizationStyle(show_frames=False, show_joints=False)
    animation = MatplotlibBackend.animate(scenes, style=style)
    try:
        animation._init_func()
        for index, scene in enumerate(scenes):
            animation._func(index)
            ax = animation._fig.axes[0]
            actual = [
                np.array(line.get_data_3d()).T for line in ax.lines
            ]
            expected = [
                np.array([seg.start, seg.end])
                for seg in scene.dh_segments
                if not np.array_equal(seg.start, seg.end)
            ]
            assert len(actual) == len(expected)
            for drawn, segment in zip(actual, expected):
                np.testing.assert_allclose(drawn, segment)
    finally:
        animation._draw_was_started = True


def test_threejs_scene_payload_orders_signed_d_a_segments():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((-2.0, sp.pi / 2, -0.4, q)), {q: 0.2})
    payload = _scene_to_payload(scene)
    segments = payload["dh_segments"]
    assert len(segments) == 2
    assert [(s["row_index"], s["kind"]) for s in segments] == [
        (1, "d"), (1, "a"),
    ]
    for original, serialized in zip(scene.dh_segments, segments):
        np.testing.assert_allclose(serialized["start"], original.start)
        np.testing.assert_allclose(serialized["end"], original.end)
    assert isinstance(json.dumps(payload), str)


def test_threejs_zero_segments_remain_in_json_for_topology():
    q = sp.Symbol("q")
    scenes = _animated_scene(
        Robot((0.5, 0, q, 0, "p")),
        [{q: 0.0}, {q: 0.3}],
    )
    payloads = _scenes_to_payload(scenes)
    assert all(len(p["dh_segments"]) == 2 for p in payloads)
    assert payloads[0]["dh_segments"][0]["start"] == (
        payloads[0]["dh_segments"][0]["end"]
    )
    assert payloads[1]["dh_segments"][0]["start"] != (
        payloads[1]["dh_segments"][0]["end"]
    )


def test_threejs_render_contains_evaluated_geometry_payload():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.4, 0, 0.3, q)), {q: 0})
    html = ThreeJSBackend.render(scene).data
    assert isinstance(ThreeJSBackend.render(scene), HTML)
    payload = _html_payload(html, "data")
    assert payload["dh_segments"] == _scene_to_payload(scene)["dh_segments"]
    assert "dhSegments.forEach(function(segment)" in html
    assert "new THREE.Vector3().fromArray(segment.start)" in html


def test_threejs_animate_contains_global_geometry_and_persistent_updates():
    q = sp.Symbol("q")
    scenes = _animated_scene(
        Robot((0.5, 0, q, 0, "p")),
        [{q: 0.0}, {q: 0.25}, {q: 0.5}],
    )
    html = ThreeJSBackend.animate(scenes).data
    payload = _html_payload(html, "animationData")
    assert all(len(frame["dh_segments"]) == 2 for frame in payload["frames"])
    assert "var linkCount = usesDH" in html
    assert "paintCylinder(robotObjects.linkMeshes[i], tmp.start, tmp.end, tmp)" in html
    assert "expectedDH !== actualDH" in html
    assert "computeLinkCorner(" in html  # Legacy only.


def test_threejs_scene_payload_legacy_contract_unchanged():
    scene = SceneData(
        [np.zeros(3)], [FrameData(np.eye(4))], 1.0
    )
    assert set(_scene_to_payload(scene)) == {"joints", "frames", "dimension"}
    html = ThreeJSBackend.render(scene).data
    assert "dh_segments" not in _html_payload(html, "data")


def test_threejs_length_cutoffs_are_unit_agnostic():
    js = _load_template_resource("threejs_common.js")
    assert "length < 1e-6" not in js
    assert "Math.abs(d) < 1e-6" not in js
    assert "length === 0" in js


def test_explicit_scene_dh_metadata_in_pure_legacy_constructor():
    scene = SceneData(
        joints=[np.zeros(3), np.array([3., 0., 0.])],
        frames=[FrameData(np.eye(4)), FrameData(np.eye(4))],
        dimension=3.0,
        dh_segments=[
            DHSegmentData(1, "d", np.zeros(3), np.array([0., 0., 2.])),
            DHSegmentData(1, "a", np.array([0., 0., 2.]),
                          np.array([3., 0., 0.])),
        ],
    )
    lines = _plot_lines(scene)
    assert len(lines) == 2
    np.testing.assert_allclose(lines[0][1], [0, 0, 2])
    assert len(_scene_to_payload(scene)["dh_segments"]) == 2
