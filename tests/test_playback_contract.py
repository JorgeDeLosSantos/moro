"""T-06 physical-time validation and backwards compatibility tests."""

import numpy as np
import pytest

from IPython.display import HTML

from moro.abc import q1
from moro.core import Robot
from moro.visualization import RobotVisualizer
from moro.visualization.playback import (
    _normalize_playback_speed,
    _normalize_playback_times,
)


@pytest.fixture
def visualizer():
    return RobotVisualizer(Robot((1.0, 0, 0, q1, "r")))


def test_nonuniform_times_are_relative_in_seconds_and_defensively_copied():
    supplied = np.array([-2.0, -1.9, -1.3, -1.2, -0.5])
    original = supplied.copy()
    result = _normalize_playback_times(supplied, 5)
    np.testing.assert_allclose(result, [0.0, 0.1, 0.7, 0.8, 1.5],
                               atol=1e-14)
    result[0] = 99.0
    np.testing.assert_array_equal(supplied, original)


def test_time_origin_shift_preserves_relative_time_and_nonuniformity():
    a = _normalize_playback_times([0, 0.1, 0.7, 0.8, 1.5], 5)
    b = _normalize_playback_times([3, 3.1, 3.7, 3.8, 4.5], 5)
    np.testing.assert_allclose(a, b, atol=1e-14)
    assert np.any(np.abs(np.diff(a) - np.diff(a)[0]) > 1e-8)


@pytest.mark.parametrize("invalid", [
    [],
    [0.0],
    0.0,
    [[0.0, 1.0]],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, np.nan, 1.0],
    [0.0, np.inf, 1.0],
    [0.0, 1j, 1.0],
    [False, 0.5, 1.0],
    [0.0, True, 1.0],
    ["0.0", "0.5", "1.0"],
    [[0.0], [0.5], [1.0]],
])
def test_reject_invalid_timestamp_shapes_types_and_order(invalid):
    with pytest.raises(ValueError, match="times"):
        _normalize_playback_times(invalid, 3)


def test_reject_timestamp_count_mismatch():
    with pytest.raises(ValueError, match="one timestamp"):
        _normalize_playback_times([0.0, 1.0], 3)


@pytest.mark.parametrize("invalid", [
    [1e16, 1e16 + 1.0, 1e16 + 4.0],
    [-1e308, 0.0, 1e308],
])
def test_reject_unrepresentable_float64_timestamp_differences(invalid):
    with pytest.raises(ValueError, match="times"):
        _normalize_playback_times(invalid, 3)


@pytest.mark.parametrize("value", [
    0, -1, np.nan, np.inf, -np.inf, True, False,
    [1.0], 1.0 + 1j, "2",
])
def test_invalid_playback_speed(value):
    with pytest.raises(ValueError, match="playback_speed"):
        _normalize_playback_speed(value)


@pytest.mark.parametrize("value", [0.25, 1.0, 2, np.float64(4.0)])
def test_valid_playback_speed(value):
    assert _normalize_playback_speed(value) == float(value)


def test_legacy_threejs_animation_has_unchanged_return_type(visualizer):
    result = visualizer.animate(
        [[0.0], [0.25]], backend="threejs", interval=50
    )
    assert isinstance(result, HTML)
    assert '"interval": 50.0' in result.data
    assert '"relative_times"' not in result.data


def test_legacy_default_and_explicit_default_speed_are_compatible(visualizer):
    data = [[0.0], [0.25]]
    first = visualizer.animate(data)
    second = visualizer.animate(data, playback_speed=1.0)
    assert isinstance(first, HTML)
    assert isinstance(second, HTML)


def test_matplotlib_legacy_unchanged(visualizer):
    from matplotlib.animation import FuncAnimation
    from matplotlib import pyplot as plt

    anim = visualizer.animate(
        [[0.0], [0.25]], backend="matplotlib", interval=40
    )
    assert isinstance(anim, FuncAnimation)
    anim._draw_was_started = True
    plt.close(anim._fig)


def test_speed_without_times_is_explicitly_rejected(visualizer):
    with pytest.raises(ValueError, match="requires explicit times"):
        visualizer.animate([[0], [1]], playback_speed=2.0)


def test_times_and_even_default_interval_conflict(visualizer):
    with pytest.raises(ValueError, match="times and interval"):
        visualizer.animate(
            [[0], [1]], times=[0.0, 1.0], interval=100,
        )


def test_explicit_times_with_matplotlib_reports_backend_limitation(visualizer):
    with pytest.raises(ValueError, match="Matplotlib"):
        visualizer.animate(
            [[0], [1]],
            times=[2.0, 3.0],
            backend="matplotlib",
        )


def test_threejs_time_mode_is_available_without_silent_uniform_fallback(visualizer):
    result = visualizer.animate(
        [[0.0], [0.25], [0.5]],
        times=[0.0, 0.1, 1.0],
        playback_speed=1.5,
    )
    assert isinstance(result, HTML)
    assert '"relative_times": [0.0, 0.1, 1.0]' in result.data
    assert '"playback_speed": 1.5' in result.data
    assert "frameIndexAtTime" in result.data
    assert "performance.now()" in result.data


def test_public_times_validation_precedes_rendering(visualizer):
    with pytest.raises(ValueError, match="strictly increasing"):
        visualizer.animate([[0], [1]], times=[1.0, 1.0])


def test_single_frame_can_use_legacy_but_not_physical_time(visualizer):
    assert isinstance(visualizer.animate([[0]]), HTML)
    with pytest.raises(ValueError, match="at least two"):
        visualizer.animate([[0]], times=[1.0])
