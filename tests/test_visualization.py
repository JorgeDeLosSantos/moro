import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    FrameData,
    MatplotlibBackend,
    RobotVisualizer,
    SceneData,
    ThreeJSBackend,
    VisualizationStyle,
    _render_html_template,
    _replace_placeholders,
    _scene_to_payload,
    _scenes_to_payload,
    _style_to_payload,
    evaluate_robot,
)
from moro.visualization.threejs_backend import _load_template_resource


@pytest.fixture
def scene_data():
    """Minimal numerical scene for backend tests."""
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
    """Two-link planar RR robot and its symbolic joint variables."""
    q1, q2 = symbols("q1 q2")

    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (1.0, 0, 0, q2, "r"),
    )

    return robot, q1, q2


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures after every test to avoid leaking Matplotlib state."""
    yield
    plt.close("all")


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
    assert "__COMMON_SCRIPT__" not in html
    assert "robot123" in html
    assert "800" in html
    assert "window.MoroThreeJS" in html


def test_load_template_resource_reads_common_script():
    common_script = _load_template_resource("threejs_common.js")

    assert "window.MoroThreeJS" in common_script
    assert "createCameras" in common_script


# ========================================
# Tests for evaluate_robot function
# ========================================

def test_evaluate_robot_returns_scene_data(simple_robot):
    robot, q1, q2 = simple_robot
    scene = evaluate_robot(robot, {q1: 0.0, q2: 0.0})

    assert isinstance(scene, SceneData)
    assert len(scene.joints) == 3  # base + 2 links
    assert len(scene.frames) == 3
    np.testing.assert_allclose(scene.joints[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(scene.joints[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(scene.joints[2], [2.0, 0.0, 0.0])


def test_evaluate_robot_with_nonzero_configuration(simple_robot):
    robot, q1, q2 = simple_robot
    scene = evaluate_robot(robot, {q1: np.pi / 2, q2: 0.0})

    assert isinstance(scene, SceneData)
    # After rotating q1 by 90°, the second joint should be at [0, 1, 0]
    np.testing.assert_allclose(scene.joints[1], [0.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(scene.joints[2], [0.0, 2.0, 0.0], atol=1e-10)


def test_evaluate_robot_dimension_scales_with_configuration(simple_robot):
    robot, q1, q2 = simple_robot
    scene = evaluate_robot(robot, {q1: 0.0, q2: 0.0})

    # max coord is 2.0, so dimension = max(2.0 * 1.5, 1.0) = 3.0
    assert scene.dimension == 3.0


# ========================================
# Tests for RobotVisualizer class
# ========================================

def test_robot_visualizer_accepts_robot(simple_robot):
    robot, _, _ = simple_robot
    viz = RobotVisualizer(robot)
    assert viz.robot is robot


def test_robot_visualizer_rejects_non_robot():
    with pytest.raises(TypeError, match="Expected a Robot instance"):
        RobotVisualizer("not a robot")


def test_robot_visualizer_plot_unknown_backend(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

    with pytest.raises(ValueError, match="Unknown backend"):
        viz.plot({q1: 0.0, q2: 0.0}, backend="unknown")


def test_robot_visualizer_animate_unknown_backend(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

    with pytest.raises(ValueError, match="Unknown backend"):
        viz.animate([{q1: 0.0, q2: 0.0}], backend="unknown")


def test_robot_visualizer_plot_matplotlib(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

    fig, ax = viz.plot({q1: 0.0, q2: 0.0}, backend="matplotlib")

    assert isinstance(fig, Figure)
    assert ax.name == "3d"

    plt.close(fig)


def test_robot_visualizer_plot_threejs(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

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
    assert "__COMMON_SCRIPT__" not in result.data
    assert "window.MoroThreeJS" in result.data


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

    plt.close(fig)


def test_matplotlib_backend_returns_figure_and_axis(scene_data):
    fig, ax = MatplotlibBackend.render(scene_data)

    assert isinstance(fig, Figure)
    assert ax.name == "3d"

    plt.close(fig)





# ---------------------------------------------------------------------------
# Payload tests
# ---------------------------------------------------------------------------

def test_scene_to_payload_contains_frame_data(scene_data):
    payload = _scene_to_payload(scene_data)

    assert len(payload["frames"]) == 2

    base_frame = payload["frames"][0]

    assert base_frame["position"] == [0.0, 0.0, 0.0]
    assert base_frame["x"] == [1.0, 0.0, 0.0]
    assert base_frame["y"] == [0.0, 1.0, 0.0]
    assert base_frame["z"] == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Matplotlib backend tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("show_links", "expected_lines"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_matplotlib_backend_link_visibility(
    scene_data,
    show_links,
    expected_lines,
):
    style = VisualizationStyle(show_links=show_links)

    _, ax = MatplotlibBackend.render(
        scene_data,
        style=style,
    )

    assert len(ax.lines) == expected_lines


def test_matplotlib_backend_animate_returns_func_animation(scene_data):
    anim = MatplotlibBackend.animate(
        [scene_data, scene_data],
        interval=50,
    )

    assert isinstance(anim, FuncAnimation)

    # Prevent Matplotlib from warning when the test destroys an animation
    # that was intentionally never displayed.
    anim._draw_was_started = True


def test_matplotlib_backend_animate_empty_list_raises():
    with pytest.raises(
        ValueError,
        match="scene_data_list must contain at least one scene",
    ):
        MatplotlibBackend.animate([])


# ---------------------------------------------------------------------------
# Three.js backend tests
# ---------------------------------------------------------------------------

def test_threejs_backend_animate_returns_html(scene_data):
    result = ThreeJSBackend.animate(
        [scene_data, scene_data],
    )

    assert isinstance(result, HTML)
    assert "__FRAMES_DATA__" not in result.data
    assert "__UNIQUE_ID__" not in result.data
    assert "__COMMON_SCRIPT__" not in result.data
    assert "window.MoroThreeJS" in result.data
    assert "THREE.Scene" in result.data


def test_threejs_backend_outputs_have_no_unresolved_placeholders(scene_data):
    render_html = ThreeJSBackend.render(scene_data).data
    animate_html = ThreeJSBackend.animate([scene_data, scene_data]).data

    assert "__" not in render_html
    assert "__" not in animate_html


def test_style_to_payload_preserves_custom_values():
    style = VisualizationStyle(
        show_frames=False,
        show_links=False,
        show_joints=True,
        show_base=False,
        show_grid=False,
        link_color="#112233",
        joint_color="#445566",
        base_color="#778899",
        frame_scale=2.5,
        joint_size=0.7,
        base_size=0.9,
        link_linewidth=4,
    )

    payload = _style_to_payload(style)

    assert payload["show_frames"] is False
    assert payload["show_links"] is False
    assert payload["show_joints"] is True
    assert payload["show_base"] is False
    assert payload["show_grid"] is False
    assert payload["link_color"] == "#112233"
    assert payload["joint_color"] == "#445566"
    assert payload["base_color"] == "#778899"
    assert payload["frame_scale"] == 2.5
    assert payload["joint_size"] == 0.7
    assert payload["base_size"] == 0.9
    assert payload["link_linewidth"] == 4


# ---------------------------------------------------------------------------
# RobotVisualizer integration tests
# ---------------------------------------------------------------------------

def test_robot_visualizer_animate_matplotlib(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

    anim = viz.animate(
        [
            {q1: 0.0, q2: 0.0},
            {q1: 0.5, q2: -0.5},
        ],
        backend="matplotlib",
    )

    assert isinstance(anim, FuncAnimation)

    # Prevent a warning caused by destroying an undisplayed animation.
    anim._draw_was_started = True


def test_robot_visualizer_animate_threejs(simple_robot):
    robot, q1, q2 = simple_robot
    viz = RobotVisualizer(robot)

    result = viz.animate(
        [
            {q1: 0.0, q2: 0.0},
            {q1: 0.5, q2: -0.5},
        ],
        backend="threejs",
    )

    assert isinstance(result, HTML)
    assert "__FRAMES_DATA__" not in result.data
    assert "__UNIQUE_ID__" not in result.data


def test_robot_visualizer_animate_empty_list_raises(simple_robot):
    robot, _, _ = simple_robot
    viz = RobotVisualizer(robot)

    with pytest.raises(
        ValueError,
        match="num_vals_list must contain at least one configuration",
    ):
        viz.animate([])


# ---------------------------------------------------------------------------
# Optional FrameData contract tests
#
# Enable these tests after implementing FrameData.__post_init__ with:
#   - conversion to a float NumPy array,
#   - defensive copying,
#   - validation of shape (4, 4).
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="Enable after FrameData validates shape in __post_init__."
)
def test_frame_data_rejects_invalid_shape():
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        FrameData(np.eye(3))


@pytest.mark.skip(
    reason="Enable after FrameData makes a defensive copy of T."
)
def test_frame_data_copies_input_matrix():
    T = np.eye(4)
    frame = FrameData(T)

    T[0, 3] = 10.0

    assert frame.position[0] == 0.0