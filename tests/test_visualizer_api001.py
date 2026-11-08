"""API-001/A2: public visualizer integration with fixed symbolic DH parameters."""

import inspect
import json
import re
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from moro import Robot
from moro.abc import l1, l2, q1, q2
from moro.visualization import RobotVisualizer
from moro.visualization.evaluation import evaluate_robot
from moro.visualization.matplotlib_backend import MatplotlibBackend
from moro.visualization.threejs_backend import ThreeJSBackend


@pytest.fixture
def symbolic_rr():
    return RobotVisualizer(
        Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))
    )


def _payload(html):
    assert isinstance(html, HTML)
    match = re.search(r"var animationData = (\{[^\n]+\});", html.data)
    assert match
    return json.loads(match.group(1))


def test_new_signatures_are_explicit_without_legacy_keyword_aliases():
    plot_signature = inspect.signature(RobotVisualizer.plot)
    animate_signature = inspect.signature(RobotVisualizer.animate)

    assert list(plot_signature.parameters) == [
        "self", "configuration", "backend", "parameters", "kwargs"
    ]
    assert list(animate_signature.parameters) == [
        "self", "configurations", "backend", "parameters",
        "times", "playback_speed", "kwargs"
    ]
    assert plot_signature.parameters["parameters"].kind == (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert animate_signature.parameters["parameters"].kind == (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert plot_signature.parameters["backend"].kind == (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


def test_named_configuration_and_parameters_plot_matplotlib(symbolic_rr):
    fig, ax = symbolic_rr.plot(
        configuration=[0.5, -0.3],
        backend="matplotlib",
        parameters={l1: 1.0, l2: 0.8},
    )
    assert fig is not None and ax.name == "3d"
    plt.close(fig)


def test_named_configuration_and_parameters_plot_threejs(symbolic_rr):
    view = symbolic_rr.plot(
        configuration={q1: 0.5, q2: -0.3},
        parameters={l1: 1.0, l2: 0.8},
    )
    assert isinstance(view, HTML)
    assert "__UNIQUE_ID__" not in view.data


def test_plot_matches_existing_full_mapping_and_does_not_forward_parameters(
    symbolic_rr,
):
    values = {l1: 1.0, l2: 0.8, q1: 0.5, q2: -0.3}
    expected = evaluate_robot(symbolic_rr.robot, values)
    constants = {l1: 1.0, l2: 0.8}
    q = np.array([0.5, -0.3])
    original = q.copy()

    with patch.object(ThreeJSBackend, "render", return_value="rendered") as render:
        result = symbolic_rr.plot(
            q, "threejs", parameters=constants, show_frames=True
        )
        assert result == "rendered"
        (actual,), kwargs = render.call_args
        np.testing.assert_allclose(
            actual.frames[-1].T, expected.frames[-1].T,
        )
        assert kwargs == {"show_frames": True}
    assert constants == {l1: 1.0, l2: 0.8}
    np.testing.assert_array_equal(q, original)


def test_legacy_positional_full_mapping_still_renders_matplotlib(symbolic_rr):
    fig, ax = symbolic_rr.plot(
        {l1: 1.0, l2: 0.8, q1: 0.5, q2: -0.3}, "matplotlib"
    )
    assert ax.name == "3d"
    plt.close(fig)


def test_named_animation_parameters_times_and_speed_threejs(symbolic_rr):
    q = np.array([
        [0.0, 0.0],
        [0.2, -0.1],
        [0.5, -0.4],
        [0.8, -0.6],
    ])
    t = np.array([0.0, 0.1, 0.7, 1.5])
    constants = {l1: 1.0, l2: 0.8}
    original_q = q.copy()
    original_t = t.copy()

    html = symbolic_rr.animate(
        configurations=q,
        parameters=constants,
        times=t,
        playback_speed=1.5,
    )
    payload = _payload(html)
    assert len(payload["frames"]) == 4
    np.testing.assert_allclose(payload["relative_times"], t)
    assert payload["playback_speed"] == 1.5
    # The backend retains the legacy interval field in its serialized
    # payload; relative_times selects physical-time mode in the viewer.
    assert payload["interval"] == 100.0
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(t, original_t)
    assert constants == {l1: 1.0, l2: 0.8}


def test_named_animation_parameters_matplotlib_legacy(symbolic_rr):
    q = [[0.0, 0.0], [0.2, -0.1], [0.4, -0.2]]
    anim = symbolic_rr.animate(
        configurations=q,
        backend="matplotlib",
        parameters={l1: 1.0, l2: 0.8},
        interval=90,
    )
    assert isinstance(anim, FuncAnimation)
    assert anim.event_source.interval == 90
    anim._draw_was_started = True
    plt.close(anim._fig)


def test_animate_full_legacy_mappings_compatible(symbolic_rr):
    q = [
        {q1: 0.0, q2: 0.0, l1: 1.0, l2: 0.8},
        {q1: 0.2, q2: -0.1, l1: 1.0, l2: 0.8},
    ]
    html = symbolic_rr.animate(q, "threejs")
    assert isinstance(html, HTML)
    assert "relative_times" not in _payload(html)


def test_partial_symbolic_frame_mappings_share_fixed_parameters(symbolic_rr):
    configs = (
        {q1: angle, q2: -angle / 2}
        for angle in (0.0, 0.2, 0.4)
    )
    with patch.object(
        MatplotlibBackend, "animate", return_value="animation"
    ) as render:
        result = symbolic_rr.animate(
            configs, "matplotlib",
            parameters={l1: 1.0, l2: 0.8},
            interval=70,
        )
        assert result == "animation"
        (scenes,), backend_kwargs = render.call_args
        assert len(scenes) == 3
        assert backend_kwargs == {"interval": 70}
        expected = evaluate_robot(
            symbolic_rr.robot,
            {q1: 0.4, q2: -0.2, l1: 1, l2: 0.8},
        )
        np.testing.assert_allclose(
            scenes[-1].frames[-1].T,
            expected.frames[-1].T,
        )


def test_rp_mixed_joint_geometry_with_constant_parameters():
    L, h = sp.symbols("L h", real=True)
    viz = RobotVisualizer(Robot(
        (L / 2, 0, 0, q1, "r"),
        (0, 0, q2 + h, 0, "p"),
    ))
    # Composed prismatic entry cannot be interpreted as a plain joint vector.
    with pytest.raises(ValueError, match="distinct atomic coordinates"):
        viz.plot([0.2, 0.3], parameters={L: 2, h: 0.4})
    html = viz.plot(
        {q1: 0.2, q2: 0.3}, parameters={L: 2, h: 0.4},
    )
    assert isinstance(html, HTML)


def test_plot_rejects_missing_unknown_and_overlapping_parameters(symbolic_rr):
    with pytest.raises(ValueError, match="Missing model parameters"):
        symbolic_rr.plot([0.0, 0.0], parameters={l1: 1})
    with pytest.raises(ValueError, match="Unknown model parameter"):
        symbolic_rr.plot(
            [0.0, 0.0],
            parameters={l1: 1, l2: 0.8, sp.Symbol("extra"): 2},
        )
    with pytest.raises(ValueError, match="Duplicate substitutions"):
        symbolic_rr.plot(
            {q1: 0.0, q2: 0.0, l1: 1.0},
            parameters={l1: 1.0, l2: 0.8},
        )


def test_animation_reports_frame_number_for_incomplete_configurations(symbolic_rr):
    with pytest.raises(ValueError, match="Frame 1: Missing joint coordinates"):
        symbolic_rr.animate(
            [{q1: 0, q2: 0}, {q1: 0.1}],
            parameters={l1: 1.0, l2: 0.8},
        )


def test_old_named_parameters_are_not_supported(symbolic_rr):
    full = {q1: 0.0, q2: 0.0, l1: 1.0, l2: 0.8}
    with pytest.raises(TypeError, match="configuration"):
        symbolic_rr.plot(num_vals=full)
    with pytest.raises(TypeError, match="configurations"):
        symbolic_rr.animate(num_vals_list=[full])


def test_matplotlib_times_limitation_preserved_with_symbolic_parameters(symbolic_rr):
    with pytest.raises(ValueError, match="Matplotlib"):
        symbolic_rr.animate(
            [[0.0, 0.0], [0.2, -0.1]],
            backend="matplotlib",
            parameters={l1: 1.0, l2: 0.8},
            times=[0.0, 1.0],
        )


def test_threejs_legacy_interval_unchanged_with_parameters(symbolic_rr):
    html = symbolic_rr.animate(
        [[0.0, 0.0], [0.1, 0.2]],
        parameters={l1: 1.0, l2: 0.8},
        interval=125,
    )
    payload = _payload(html)
    assert payload["interval"] == 125.0
    assert "relative_times" not in payload
