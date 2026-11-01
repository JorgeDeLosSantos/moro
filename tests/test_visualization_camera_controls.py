"""V5-C1 contracts: familiar orbit gestures, Right view and safe Fit View.

The browser interaction is implemented in the shared Three.js helper.
These tests guard both emitted HTML templates and the reusable camera
control contract; browser-level geometry is checked separately.
"""

from importlib.resources import files

import sympy as sp
from moro.core import Robot
from moro.visualization import ThreeJSBackend, evaluate_robot


def _source(name: str) -> str:
    return (
        files("moro.visualization.templates")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def _scenes():
    q = sp.Symbol("q")
    robot = Robot((0.5, sp.pi / 2, 0.25, q))
    return [
        evaluate_robot(robot, {q: 0.1}),
        evaluate_robot(robot, {q: 0.8}),
    ]


def test_shared_orbitcontrols_uses_standard_mouse_and_touch_gestures():
    src = _source("threejs_common.js")
    assert "LEFT: THREE.MOUSE.ROTATE" in src
    assert "MIDDLE: THREE.MOUSE.DOLLY" in src
    assert "RIGHT: THREE.MOUSE.PAN" in src
    assert "ONE: THREE.TOUCH.ROTATE" in src
    assert "TWO: THREE.TOUCH.DOLLY_PAN" in src
    assert "LEFT: THREE.MOUSE.PAN" not in src
    assert "MIDDLE: THREE.MOUSE.ROTATE" not in src


def test_right_preset_looks_from_positive_x_with_positive_z_up():
    src = _source("threejs_common.js")
    right = src.split("case 'right':", 1)[1].split("case 'top':", 1)[0]
    assert "camera.up.set(0, 0, 1)" in right
    assert "target.x + distance" in right
    assert "target.y," in right
    assert "target.z" in right
    assert "break;" in right
    # Front and Top presets remain available and are unaffected.
    assert "case 'front':" in src
    assert "case 'top':" in src
    assert "case 'isometric':" in src


def test_fit_helper_recenters_preserving_active_orientation_and_projection():
    src = _source("threejs_common.js")
    helper = src.split("function fitCameraView(options)", 1)[1].split(
        "function createCameras(options)", 1
    )[0]
    assert "subVectors(" in helper
    assert "camera.position, controls.target" in helper
    assert "resetCameraFraming({" in helper
    assert "camera.position.copy(target).addScaledVector(offset, fit.distance)" in helper
    assert "controls.target.copy(target)" in helper
    assert "camera.lookAt(target)" in helper
    assert "controls.update()" in helper
    assert "setPresetView" not in helper
    assert "controls.object =" not in helper
    assert "camera.up.set(" not in helper


def test_static_toolbar_offers_right_and_fit_view_without_reset_action():
    html = _source("threejs_viewer.html")
    assert "setView('front')" in html
    assert "setView('right')" in html
    assert "setView('top')" in html
    assert "setView('isometric')" in html
    assert "Fit View" in html
    assert ".fitView()" in html
    assert "function fitView()" in html
    assert "window.MoroThreeJS.fitCameraView({" in html
    assert "function resetView()" not in html
    assert "Reset View" not in html
    assert "Left drag to orbit" in html
    assert "Right drag to pan" in html
    assert "Wheel/pinch to zoom" in html


def test_animation_toolbar_offers_right_and_fit_view_without_reset_action():
    html = _source("threejs_animation.html")
    assert "setView('right')" in html
    assert "Fit View" in html
    assert ".fitView()" in html
    assert "function fitView()" in html
    assert "window.MoroThreeJS.fitCameraView({" in html
    assert "function resetView()" not in html
    assert "&#9654; Play" in html
    assert "function goToFrame(index)" in html
    assert "function togglePlay()" in html
    assert "Left drag to orbit" in html
    assert "Right drag to pan" in html


def test_fit_view_on_animation_cannot_rewind_or_pause():
    src = _source("threejs_animation.html")
    implementation = src.split("function fitView()", 1)[1].split(
        "function animate()", 1
    )[0]
    assert "window.MoroThreeJS.fitCameraView({" in implementation
    assert "goToFrame(" not in implementation
    assert "pausePlayback(" not in implementation
    assert "setView(" not in implementation
    assert "currentFrame =" not in implementation
    assert "isPlaying =" not in implementation
    assert "playInterval =" not in implementation


def test_emitted_html_includes_shared_camera_controls():
    scenes = _scenes()
    html_static = ThreeJSBackend.render(scenes[0]).data
    html_animation = ThreeJSBackend.animate(scenes).data
    for html in (html_static, html_animation):
        assert "function fitCameraView(options)" in html
        assert "setView('right')" in html
        assert "Fit View" in html
        assert "LEFT: THREE.MOUSE.ROTATE" in html
        assert "ONE: THREE.TOUCH.ROTATE" in html
        assert "MIDDLE: THREE.MOUSE.ROTATE" not in html
        assert "__COMMON_SCRIPT__" not in html
        assert "__UNIQUE_ID__" not in html
