from matplotlib.figure import Figure
import numpy as np
import pytest

from IPython.display import HTML
from moro.visualization import (
    FrameData,
    SceneData,
    ThreeJSBackend,
    VisualizationStyle,
    MatplotlibBackend,
    _scene_to_payload,
    _scenes_to_payload,
    _replace_placeholders,
    evaluate_robot,
    RobotVisualizer,
)
import json
from moro.visualization import _render_html_template
from moro.core import Robot
from sympy import symbols


@pytest.fixture
def scene_data():
    T0 = np.eye(4)

    T1 = np.eye(4)
    T1[:3, 3] = [1.0, 0.0, 0.0]

    return SceneData(
        joints=[
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ],
        frames=[
            FrameData(T0),
            FrameData(T1),
        ],
        dimension=1.5,
    )


@pytest.fixture
def simple_robot():
    q1, q2 = symbols("q1 q2")
    l1, l2 = 1.0, 1.0
    return Robot((l1, 0, 0, q1), (l2, 0, 0, q2))


# ========================================
# Tests for FrameData class
# ========================================

def test_frame_data_extracts_position_and_rotation():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]

    frame = FrameData(T)

    np.testing.assert_allclose(
        frame.position,
        [1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(
        frame.rotation,
        np.eye(3),
    )


def test_frame_data_extracts_axes():
    T = np.eye(4)
    frame = FrameData(T)

    np.testing.assert_allclose(frame.x, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(frame.y, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(frame.z, [0.0, 0.0, 1.0])


def test_frame_data_with_rotation():
    """FrameData axes should reflect a non-identity rotation."""
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [1.0, 2.0, 3.0]

    frame = FrameData(T)

    np.testing.assert_allclose(frame.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame.rotation, R)
    np.testing.assert_allclose(frame.x, R[:, 0])
    np.testing.assert_allclose(frame.y, R[:, 1])
    np.testing.assert_allclose(frame.z, R[:, 2])


# ========================================
# Tests for VisualizationStyle class
# =======================================

def test_visualization_style_defaults():
    style = VisualizationStyle()

    assert style.show_frames is True
    assert style.show_links is True
    assert style.show_joints is True
    assert style.show_base is True
    assert style.show_grid is True

    assert style.frame_scale is None
    assert style.joint_size is None
    assert style.base_size is None


def test_visualization_style_can_be_customized():
    style = VisualizationStyle(
        show_frames=False,
        link_color="#000000",
        link_linewidth=5,
    )

    assert style.show_frames is False
    assert style.link_color == "#000000"
    assert style.link_linewidth == 5


def test_visualization_style_all_visibility_flags():
    """All visibility flags can be set to False."""
    style = VisualizationStyle(
        show_frames=False,
        show_links=False,
        show_joints=False,
        show_base=False,
        show_grid=False,
    )

    assert style.show_frames is False
    assert style.show_links is False
    assert style.show_joints is False
    assert style.show_base is False
    assert style.show_grid is False


# ========================================
# Tests for _scene_to_payload function
# ======================================

def test_scene_to_payload_is_json_serializable():
    scene = SceneData(
        joints=[
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
        ],
        frames=[
            FrameData(np.eye(4)),
            FrameData(np.eye(4)),
        ],
        dimension=3.0,
    )

    payload = _scene_to_payload(scene)

    result = json.dumps(payload)

    assert isinstance(result, str)
    assert payload["dimension"] == 3.0
    assert payload["joints"][1] == [1.0, 2.0, 3.0]


# ========================================
# Tests for _scenes_to_payload function
# ========================================

def test_scenes_to_payload_converts_multiple_scenes():
    scene1 = SceneData(
        joints=[np.array([0.0, 0.0, 0.0])],
        frames=[FrameData(np.eye(4))],
        dimension=1.0,
    )
    scene2 = SceneData(
        joints=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        frames=[FrameData(np.eye(4)), FrameData(np.eye(4))],
        dimension=2.0,
    )

    payloads = _scenes_to_payload([scene1, scene2])

    assert len(payloads) == 2
    assert payloads[0]["dimension"] == 1.0
    assert payloads[1]["dimension"] == 2.0
    assert len(payloads[0]["joints"]) == 1
    assert len(payloads[1]["joints"]) == 2


# ========================================
# Tests for _replace_placeholders function
# ========================================

def test_replace_placeholders_replaces_all_tokens():
    template = '<div id="__UNIQUE_ID__" style="width:__WIDTH__px;"></div>'
    result = _replace_placeholders(template, {"unique_id": "abc123", "width": 800})

    assert "__UNIQUE_ID__" not in result
    assert "__WIDTH__" not in result
    assert "abc123" in result
    assert "800" in result


def test_replace_placeholders_rejects_unknown_replacement():
    template = '<div id="__UNIQUE_ID__"></div>'

    with pytest.raises(ValueError, match="Placeholder '__MISSING_VALUE__' was not found"):
        _replace_placeholders(template, {"missing_value": 100})


def test_replace_placeholders_raises_on_unresolved_placeholders():
    template = '<div id="__UNIQUE_ID__"><span>__UNRESOLVED__</span></div>'

    with pytest.raises(ValueError, match="Unresolved placeholders"):
        _replace_placeholders(template, {"unique_id": "abc"})


# ========================================
# Tests for _render_html_template function
# ========================================

def test_render_html_template_replaces_placeholders():
    html = _render_html_template(
        "threejs_viewer.html",
        {
            "unique_id": "robot123",
            "width": 800,
            "height": 600,
            "robot_data": '{"test": true}',
        },
    )

    assert "__UNIQUE_ID__" not in html
    assert "__WIDTH__" not in html
    assert "__HEIGHT__" not in html
    assert "__ROBOT_DATA__" not in html
    assert "robot123" in html
    assert "800" in html


# ========================================
# Tests for evaluate_robot function
# ========================================

def test_evaluate_robot_returns_scene_data(simple_robot):
    q1, q2 = symbols("q1 q2")
    scene = evaluate_robot(simple_robot, {q1: 0.0, q2: 0.0})

    assert isinstance(scene, SceneData)
    assert len(scene.joints) == 3  # base + 2 links
    assert len(scene.frames) == 3
    np.testing.assert_allclose(scene.joints[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(scene.joints[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(scene.joints[2], [2.0, 0.0, 0.0])


def test_evaluate_robot_with_nonzero_configuration(simple_robot):
    q1, q2 = symbols("q1 q2")
    scene = evaluate_robot(simple_robot, {q1: np.pi / 2, q2: 0.0})

    assert isinstance(scene, SceneData)
    # After rotating q1 by 90°, the second joint should be at [0, 1, 0]
    np.testing.assert_allclose(scene.joints[1], [0.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(scene.joints[2], [0.0, 2.0, 0.0], atol=1e-10)


def test_evaluate_robot_dimension_scales_with_configuration(simple_robot):
    q1, q2 = symbols("q1 q2")
    scene = evaluate_robot(simple_robot, {q1: 0.0, q2: 0.0})

    # max coord is 2.0, so dimension = max(2.0 * 1.5, 1.0) = 3.0
    assert scene.dimension == 3.0


# ========================================
# Tests for RobotVisualizer class
# ========================================

def test_robot_visualizer_accepts_robot(simple_robot):
    viz = RobotVisualizer(simple_robot)
    assert viz.robot is simple_robot


def test_robot_visualizer_rejects_non_robot():
    with pytest.raises(TypeError, match="Expected a Robot instance"):
        RobotVisualizer("not a robot")


def test_robot_visualizer_plot_unknown_backend(simple_robot):
    q1, q2 = symbols("q1 q2")
    viz = RobotVisualizer(simple_robot)

    with pytest.raises(ValueError, match="Unknown backend"):
        viz.plot({q1: 0.0, q2: 0.0}, backend="unknown")


def test_robot_visualizer_animate_unknown_backend(simple_robot):
    q1, q2 = symbols("q1 q2")
    viz = RobotVisualizer(simple_robot)

    with pytest.raises(ValueError, match="Unknown backend"):
        viz.animate([{q1: 0.0, q2: 0.0}], backend="unknown")


def test_robot_visualizer_plot_matplotlib(simple_robot):
    q1, q2 = symbols("q1 q2")
    viz = RobotVisualizer(simple_robot)

    fig, ax = viz.plot({q1: 0.0, q2: 0.0}, backend="matplotlib")

    assert isinstance(fig, Figure)
    assert ax.name == "3d"


def test_robot_visualizer_plot_threejs(simple_robot):
    q1, q2 = symbols("q1 q2")
    viz = RobotVisualizer(simple_robot)

    result = viz.plot({q1: 0.0, q2: 0.0}, backend="threejs")

    assert isinstance(result, HTML)


# ========================================
# Rendering Backend Tests
# ========================================

def test_threejs_backend_returns_html(scene_data):
    result = ThreeJSBackend.render(scene_data)

    assert isinstance(result, HTML)
    assert "THREE.Scene" in result.data
    assert "__ROBOT_DATA__" not in result.data
    assert "__UNIQUE_ID__" not in result.data


def test_threejs_backend_animate_empty_list_raises():
    with pytest.raises(ValueError, match="scene_data_list must contain at least one scene"):
        ThreeJSBackend.animate([])


def test_matplotlib_backend_can_hide_links(scene_data):
    style = VisualizationStyle(show_links=False)

    fig, ax = MatplotlibBackend.render(
        scene_data,
        style=style,
    )

    assert len(ax.lines) == 0


def test_matplotlib_backend_returns_figure_and_axis(scene_data):
    fig, ax = MatplotlibBackend.render(scene_data)

    assert isinstance(fig, Figure)
    assert ax.name == "3d"