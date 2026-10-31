"""Moro 0.6.0 V4-B: rendering adapters for common scene framing."""

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
    FrameData, MatplotlibBackend, SceneData, ThreeJSBackend,
    VisualizationStyle, evaluate_robot, resolve_framing, resolve_layout,
)
from moro.visualization.threejs_backend import _framing_to_payload


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _extract_html_payload(html, variable):
    match = re.search(r"var " + variable + r" = (\{[^\n]+\});", html)
    assert match is not None
    return json.loads(match.group(1))


def _legacy_offset_scene(start=100.0):
    T0 = np.eye(4)
    T1 = np.eye(4)
    T0[:3, 3] = (start, -25, 8)
    T1[:3, 3] = (start + 4, -22, 10)
    return SceneData(
        joints=[T0[:3, 3].copy(), T1[:3, 3].copy()],
        frames=[FrameData(T0), FrameData(T1)],
        dimension=10000.0,
    )


def _limits(ax):
    return (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())


def test_matplotlib_static_limits_center_on_geometry_not_origin():
    scene = _legacy_offset_scene()
    style = VisualizationStyle(show_frames=False, show_joints=False)
    layout = resolve_layout([scene], reference_length=10.0)
    framing = resolve_framing([scene], layout=layout, style=style)

    _, ax = MatplotlibBackend.render(scene, style=style, layout=layout)
    # Standalone render calculates the same framing using supplied layout.
    for (low, high), center in zip(_limits(ax), framing.center):
        assert (low + high) / 2 == pytest.approx(center)
        assert high - low == pytest.approx(2 * framing.half_extent)
    assert ax.get_box_aspect() == pytest.approx(
        (ax.get_box_aspect()[0],) * 3
    )


def test_matplotlib_explicit_framing_controls_view_and_marker_conversion():
    scene = _legacy_offset_scene()
    layout = resolve_layout([scene], reference_length=10)
    style = VisualizationStyle(show_frames=False)
    framing = resolve_framing([scene], layout=layout, style=style)

    _, ax = MatplotlibBackend.render(
        scene, layout=layout, framing=framing, style=style,
    )
    for i, limits in enumerate(_limits(ax)):
        assert limits == pytest.approx((
            framing.center[i] - framing.half_extent,
            framing.center[i] + framing.half_extent,
        ))

    sizes = [collection.get_sizes()[0] for collection in ax.collections]
    assert len(sizes) == 2

    wider = resolve_framing(
        [scene], layout=resolve_layout([scene], reference_length=100),
        style=style,
    )
    # Using a wider explicit framing reduces screen-space nominal marker area
    # while preserving the exact layout sizes.
    from dataclasses import replace
    wider = replace(wider, half_extent=framing.half_extent * 2)
    _, ax2 = MatplotlibBackend.render(
        scene, layout=layout, framing=wider, style=style,
    )
    sizes_wider = [collection.get_sizes()[0] for collection in ax2.collections]
    assert sizes_wider == pytest.approx([value / 4 for value in sizes])


def test_matplotlib_animated_limits_are_stable_after_redraw():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: value}) for value in (0, 0.2, 0.7, 0.1)]
    layout = resolve_layout(scenes)
    style = VisualizationStyle(show_trajectory=True)
    framing = resolve_framing(scenes, layout=layout, style=style)
    anim = MatplotlibBackend.animate(scenes, layout=layout, style=style)
    try:
        anim._init_func()
        expected = None
        for i in range(len(scenes)):
            anim._func(i)
            ax = anim._fig.axes[0]
            actual = _limits(ax)
            for j in range(3):
                assert actual[j] == pytest.approx((
                    framing.center[j] - framing.half_extent,
                    framing.center[j] + framing.half_extent,
                ))
            if expected is not None:
                assert actual == expected
            expected = actual
    finally:
        anim._draw_was_started = True


def test_matplotlib_world_scale_units_remain_equivalent():
    q = symbols("q")
    sizes = []
    for length in (0.25, 250.0):
        scene = evaluate_robot(Robot((length, 0, 0, q)), {q: 0})
        _, ax = MatplotlibBackend.render(
            scene, style=VisualizationStyle(show_frames=False),
        )
        sizes.append([x.get_sizes()[0] for x in ax.collections])
        plt.close(ax.figure)
    np.testing.assert_allclose(sizes[0], sizes[1])


def test_threejs_render_has_one_shared_framing_payload():
    scene = _legacy_offset_scene()
    style = VisualizationStyle(show_frames=False, show_joints=False)
    data = _extract_html_payload(
        ThreeJSBackend.render(scene, style=style).data, "data"
    )
    framing = resolve_framing([scene], style=style)
    assert data["framing"]["center"] == pytest.approx(framing.center)
    assert data["framing"]["fit_radius"] == pytest.approx(framing.fit_radius)
    assert data["framing"]["minimum"] == pytest.approx(framing.bounds.minimum)
    assert data["framing"]["maximum"] == pytest.approx(framing.bounds.maximum)
    assert set(_framing_to_payload(framing)) == {
        "minimum", "maximum", "center", "half_extent", "fit_radius",
    }


def test_threejs_animation_has_global_frame_not_per_scene():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: value}) for value in (0, .1, .9, .3)]
    data = _extract_html_payload(
        ThreeJSBackend.animate(scenes).data, "animationData"
    )
    expected = resolve_framing(scenes)
    assert data["framing"]["center"] == pytest.approx(expected.center)
    assert data["framing"]["fit_radius"] == pytest.approx(expected.fit_radius)
    assert all("framing" not in scene for scene in data["frames"])


def test_threejs_uses_camera_fit_instead_of_legacy_dimension():
    scene = _legacy_offset_scene()
    static = ThreeJSBackend.render(scene).data
    animated = ThreeJSBackend.animate([scene, scene]).data

    for html in (static, animated):
        assert "framing: framing" in html
        assert "var cameraDistance;" in html
        assert "window.MoroThreeJS.resetCameraFraming" in html
        assert "distance: cameraDistance" in html
        assert "topUp: [0, 1, 0]" in html
        assert "function resolveCameraFit(framing, aspect, fovDegrees)" in html

    assert "computeAnimationBounds(" not in animated
    assert "var nearPlane = Math.max(dim * 0.001, 0.01)" not in animated
    assert "sceneScale = Math.max(Number(data.dimension)" not in static


def test_threejs_framing_is_unit_covariant_without_absolute_floors():
    q = symbols("q")
    frames = []
    for factor in (1, 1000):
        scene = evaluate_robot(
            Robot((.25 * factor, 0, .1 * factor, q)), {q: 0}
        )
        data = _extract_html_payload(
            ThreeJSBackend.render(scene).data, "data"
        )
        frames.append(data["framing"])
    for key in ("fit_radius", "half_extent"):
        assert frames[1][key] == pytest.approx(1000 * frames[0][key])
    for key in ("minimum", "maximum", "center"):
        assert frames[1][key] == pytest.approx(
            [1000 * x for x in frames[0][key]]
        )


def test_threejs_explicit_framing_is_preserved():
    scene = _legacy_offset_scene()
    layout = resolve_layout([scene], reference_length=15)
    framing = resolve_framing([scene], layout=layout)
    payload = _extract_html_payload(
        ThreeJSBackend.render(scene, layout=layout, framing=framing).data,
        "data",
    )
    assert payload["layout"]["reference_length"] == 15
    assert payload["framing"]["fit_radius"] == pytest.approx(framing.fit_radius)
