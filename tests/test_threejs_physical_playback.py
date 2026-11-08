"""Three.js physical-time playback payload and JavaScript smoke tests.

Python tests verify serialization and compatibility; optional Node checks
exercise the actual in-template timestamp-index helper without requiring a
browser, Three.js or a Node dependency for importing Moro.
"""

import json
import re
import shutil
import subprocess
from importlib.resources import files

import numpy as np
import pytest
from IPython.display import HTML

from moro.abc import q1
from moro.core import Robot
from moro.trajectory import joint_trapezoidal_trajectory, joint_waypoint_trajectory
from moro.visualization import RobotVisualizer


@pytest.fixture
def viz():
    return RobotVisualizer(Robot((1.0, 0, 0, q1, "r")))


def _animation_payload(html):
    assert isinstance(html, HTML)
    match = re.search(r"var animationData = (\{[^\n]+\});", html.data)
    assert match, "HTML must include a serialized animation payload."
    return json.loads(match.group(1))


def test_irregular_physical_timestamps_are_passed_as_relative_seconds(viz):
    times = np.array([-2.0, -1.9, -1.3, -1.2, -0.5])
    original = times.copy()
    html = viz.animate(
        np.array([[0.0], [0.2], [0.4], [0.6], [0.8]]),
        times=times,
        playback_speed=2.0,
    )
    payload = _animation_payload(html)

    np.testing.assert_allclose(
        payload["relative_times"],
        [0.0, 0.1, 0.7, 0.8, 1.5],
        rtol=0.0, atol=1e-14,
    )
    assert payload["playback_speed"] == 2.0
    assert len(payload["frames"]) == times.size
    np.testing.assert_array_equal(times, original)
    assert "time-control-" in html.data
    assert "frameIndexAtTime" in html.data


def test_same_timestamps_with_shifted_time_origin_have_same_payload(viz):
    configs = [[0.0], [0.1], [0.5]]
    a = _animation_payload(viz.animate(configs, times=[0.0, 0.4, 1.3]))
    b = _animation_payload(viz.animate(configs, times=[10.0, 10.4, 11.3]))
    np.testing.assert_allclose(a["relative_times"], b["relative_times"], atol=1e-14)
    assert a["playback_speed"] == b["playback_speed"] == 1.0


def test_legacy_mode_retains_interval_payload_and_no_relative_time(viz):
    legacy = _animation_payload(viz.animate([[0.0], [0.5]], interval=25))
    assert legacy["interval"] == 25.0
    assert "relative_times" not in legacy
    assert "playback_speed" not in legacy

    default = _animation_payload(viz.animate([[0.0], [0.5]]))
    assert default["interval"] == 100.0
    assert "relative_times" not in default


def test_trajectory_arrays_interoperate_without_input_class_dependency(viz):
    traj = joint_trapezoidal_trajectory(
        0.0, 1.0, v_max=0.5, a_max=1.0, samples=17,
    )
    result = viz.animate(traj.q, times=traj.t)
    payload = _animation_payload(result)
    assert len(payload["frames"]) == traj.samples
    assert payload["relative_times"][-1] == pytest.approx(traj.duration)


def test_timing_is_per_widget_and_uses_monotonic_browser_clock(viz):
    html1 = viz.animate([[0.0], [0.5]], times=[0.0, 1.0]).data
    html2 = viz.animate([[0.0], [0.5]], times=[0.0, 1.0]).data
    id1 = re.search(r"id=\"controls-([0-9a-f]+)\"", html1)
    id2 = re.search(r"id=\"controls-([0-9a-f]+)\"", html2)
    assert id1 is not None and id2 is not None
    assert id1.group(1) != id2.group(1)
    for page in (html1, html2):
        assert "performance.now()" in page
        assert "syncPhysicalPlayback" in page
        assert "frameIndexAtTime" in page
        assert "anchorPhysicalElapsed" in page
        assert "timeLabel.textContent" in page
        # Legacy setInterval is still present, behind the mode switch.
        assert "window.setInterval" in page
        assert "if (hasPhysicalTime)" in page


def test_matplotlib_times_remain_explicitly_unsupported(viz):
    with pytest.raises(ValueError, match="Three.js"):
        viz.animate([[0.0], [0.5]], times=[0.0, 1.0],
                    backend="matplotlib")


def test_legacy_speed_without_times_still_rejected(viz):
    with pytest.raises(ValueError, match="requires explicit times"):
        viz.animate([[0.0], [0.5]], playback_speed=2.0)


def test_time_mode_and_explicit_interval_still_conflict(viz):
    with pytest.raises(ValueError, match="times and interval"):
        viz.animate([[0.0], [0.5]], times=[0, 1], interval=100)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_browser_timestamp_lookup_boundaries_with_node():
    """Run the actual JS binary-search helper using Node's built-in runtime."""
    raw = (
        files("moro.visualization.templates")
        .joinpath("threejs_animation.html")
        .read_text(encoding="utf-8")
    )
    opening = "            function frameIndexAtTime(elapsedSeconds) {"
    closing = "            function stopPhysicalPlayback() {"
    assert opening in raw and closing in raw
    body = raw.split(opening, 1)[1].split(closing, 1)[0]
    source = "function frameIndexAtTime(elapsedSeconds) {" + body
    script = (
        "var relativeTimes = [0, 0.1, 0.7, 0.8, 1.5];\n"
        + source
        + "\nconsole.log(JSON.stringify("
        + "[-1, 0, 0.09, 0.1, 0.4, 0.7, 0.75, 0.8, 1.49, 1.5, 2]"
        + ".map(frameIndexAtTime)));"
    )
    completed = subprocess.run(
        ["node", "-e", script],
        check=True, capture_output=True, text=True, timeout=10,
    )
    assert json.loads(completed.stdout) == [
        0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_browser_animation_template_javascript_syntax_with_node(tmp_path):
    raw = (
        files("moro.visualization.templates")
        .joinpath("threejs_animation.html")
        .read_text(encoding="utf-8")
    )
    scripts = re.findall(r"<script>(.*?)</script>", raw, flags=re.DOTALL)
    assert len(scripts) == 2
    script_file = tmp_path / "animation.js"
    script_file.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run(
        ["node", "--check", str(script_file)],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_browser_pause_resume_seek_finish_and_restart_with_node():
    """Exercise the real template functions with a synthetic monotonic clock."""
    raw = (
        files("moro.visualization.templates")
        .joinpath("threejs_animation.html")
        .read_text(encoding="utf-8")
    )
    opening = "            function displayFrame(index) {"
    closing = "            function setView(viewName) {"
    assert opening in raw and closing in raw
    state_machine = opening + raw.split(opening, 1)[1].split(closing, 1)[0]
    prelude = """
        var now = 0;
        var performance = {now: function() { return now; }};
        var relativeTimes = [0, 0.1, 0.7, 0.8, 1.5];
        var framesData = [{}, {}, {}, {}, {}];
        var hasPhysicalTime = true;
        var playbackSpeed = 1;
        var physicalDuration = 1.5;
        var physicalElapsed = 0;
        var anchorPhysicalElapsed = 0;
        var anchorWallMs = 0;
        var currentFrame = 0;
        var isPlaying = false;
        var playInterval = null;
        var slider = {value: 0};
        var frameLabel = {textContent: '0'};
        var timeLabel = {textContent: ''};
        var playButton = {innerHTML: '', setAttribute: function() {}};
        var updateRobotObjects = function() {};
        var updateTrajectoryForFrame = function() {};
        var animationData = {interval: 100};
        var window = {
            setInterval: function() {throw Error('legacy timer must not run');},
            clearInterval: function() {throw Error('legacy timer must not run');}
        };
        function snapshot() {
            return [currentFrame, isPlaying, Number(physicalElapsed.toFixed(3))];
        }
    """
    scenario = """
        var states = [];
        startPlayback();
        now = 400;
        syncPhysicalPlayback(now);
        states.push(snapshot());    // frame 1 at 0.4 s
        pausePlayback();
        now = 5000;
        syncPhysicalPlayback(now);
        states.push(snapshot());    // paused at 0.4 s
        startPlayback();            // resume from 0.4
        now = 5400;
        syncPhysicalPlayback(now);
        states.push(snapshot());    // frame 3 at 0.8 s
        goToFrame(2);               // seek backwards to 0.7 s
        states.push(snapshot());
        now = 6300;
        syncPhysicalPlayback(now);  // reaches the final frame and stops
        states.push(snapshot());
        startPlayback();            // completed playback restarts
        states.push(snapshot());
        console.log(JSON.stringify(states));
    """
    completed = subprocess.run(
        ["node", "-e", prelude + state_machine + scenario],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert json.loads(completed.stdout) == [
        [1, True, 0.4],
        [1, False, 0.4],
        [3, True, 0.8],
        [2, True, 0.7],
        [4, False, 1.5],
        [0, True, 0.0],
    ]


def test_documented_irregular_rr_waypoint_example_payload():
    """The Jupyter teaching example must work with its actual sample grid."""
    from moro.abc import q2

    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.8, 0, 0, q2, "r"),
    )
    visualizer = RobotVisualizer(robot)

    waypoints = np.array([
        [0.0, 0.0],
        [0.65, -0.3],
        [1.2, 0.4],
    ])
    sample_times = np.array([
        0.0, 0.07, 0.20, 0.50, 1.0,
        1.05, 1.40, 1.70, 2.00, 2.30, 2.60,
    ])
    motion = joint_waypoint_trajectory(
        waypoints,
        [0.0, 1.0, 2.6],
        method="cubic",
        derivative_policy="shape_preserving",
        sample_times=sample_times,
    )
    assert motion.q.shape == (11, 2)
    np.testing.assert_array_equal(motion.t, sample_times)

    result = visualizer.animate(motion.q, times=motion.t)
    payload = _animation_payload(result)
    assert len(payload["frames"]) == len(sample_times)
    np.testing.assert_allclose(
        payload["relative_times"], sample_times, atol=1e-14,
    )
    assert payload["playback_speed"] == 1.0
    assert payload["relative_times"][-1] == pytest.approx(2.6)

    legacy = _animation_payload(visualizer.animate(motion.q, interval=150))
    assert legacy["interval"] == 150.0
    assert "relative_times" not in legacy


def test_documented_rr_trapezoidal_example_has_physical_duration():
    from moro.abc import q2

    visualizer = RobotVisualizer(Robot(
        (1.0, 0, 0, q1, "r"),
        (0.8, 0, 0, q2, "r"),
    ))
    trapezoidal = joint_trapezoidal_trajectory(
        [0.0, 0.0], [1.2, -0.8],
        v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        samples=161,
    )
    assert np.isclose(trapezoidal.duration, 2.6)
    html = visualizer.animate(
        trapezoidal.q, times=trapezoidal.t, playback_speed=0.75
    )
    payload = _animation_payload(html)
    assert len(payload["frames"]) == 161
    assert payload["relative_times"][-1] == pytest.approx(
        trapezoidal.duration
    )
    assert payload["playback_speed"] == 0.75


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_browser_variable_speed_seek_when_paused_and_delayed_frame_skip():
    """Exercise actual C2 state functions under 2x time and stalled rendering."""
    raw = (
        files("moro.visualization.templates")
        .joinpath("threejs_animation.html")
        .read_text(encoding="utf-8")
    )
    opening = "            function displayFrame(index) {"
    closing = "            function setView(viewName) {"
    assert opening in raw and closing in raw
    state_machine = opening + raw.split(opening, 1)[1].split(closing, 1)[0]
    prelude = """
        var now = 0;
        var performance = {now: function() { return now; }};
        var relativeTimes = [0, 0.1, 0.7, 0.8, 1.5];
        var framesData = [{}, {}, {}, {}, {}];
        var hasPhysicalTime = true;
        var playbackSpeed = 2;
        var physicalDuration = 1.5;
        var physicalElapsed = 0;
        var anchorPhysicalElapsed = 0;
        var anchorWallMs = 0;
        var currentFrame = 0;
        var isPlaying = false;
        var playInterval = null;
        var slider = {value: 0};
        var frameLabel = {textContent: '0'};
        var timeLabel = {textContent: ''};
        var playButton = {innerHTML: '', setAttribute: function() {}};
        var updateRobotObjects = function() {};
        var updateTrajectoryForFrame = function() {};
        var animationData = {interval: 100};
        var window = {
            setInterval: function() {throw Error('legacy timer must not run');},
            clearInterval: function() {throw Error('legacy timer must not run');}
        };
        function snapshot() {
            return [currentFrame, isPlaying, Number(physicalElapsed.toFixed(3))];
        }
    """
    scenario = """
        var result = [];
        startPlayback();
        now = 200;
        syncPhysicalPlayback(now);    // 0.4 seconds of progress at 2x
        result.push(snapshot());
        now = 400;
        syncPhysicalPlayback(now);    // 0.8 seconds, frame 2 may be skipped
        result.push(snapshot());
        pausePlayback();
        goToFrame(1);                 // snap to 0.1 while paused
        result.push(snapshot());
        now = 5000;                   // long pause should not advance
        syncPhysicalPlayback(now);
        result.push(snapshot());
        startPlayback();              // resume from 0.1
        now = 5350;                   // 0.7 seconds later at double speed
        syncPhysicalPlayback(now);    // progress 0.8 seconds
        result.push(snapshot());
        now = 10000;                  // delayed tab: finish at final frame
        syncPhysicalPlayback(now);
        result.push(snapshot());
        console.log(JSON.stringify(result));
    """
    completed = subprocess.run(
        ["node", "-e", prelude + state_machine + scenario],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert json.loads(completed.stdout) == [
        [1, True, 0.4],
        [3, True, 0.8],
        [1, False, 0.1],
        [1, False, 0.1],
        [3, True, 0.8],
        [4, False, 1.5],
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_browser_legacy_uniform_player_still_loops_and_clears_timer():
    """Exercise the existing setInterval loop, not only its payload."""
    raw = (
        files("moro.visualization.templates")
        .joinpath("threejs_animation.html")
        .read_text(encoding="utf-8")
    )
    opening = "            function displayFrame(index) {"
    closing = "            function setView(viewName) {"
    state_machine = opening + raw.split(opening, 1)[1].split(closing, 1)[0]
    prelude = """
        var now = 0;
        var performance = {now: function() { return now; }};
        var relativeTimes = null;
        var framesData = [{}, {}, {}];
        var hasPhysicalTime = false;
        var playbackSpeed = 1;
        var physicalDuration = 0;
        var physicalElapsed = 0;
        var anchorPhysicalElapsed = 0;
        var anchorWallMs = 0;
        var currentFrame = 0;
        var isPlaying = false;
        var playInterval = null;
        var slider = {value: 0};
        var frameLabel = {textContent: '0'};
        var timeLabel = {textContent: ''};
        var playButton = {innerHTML: '', setAttribute: function() {}};
        var updateRobotObjects = function() {};
        var updateTrajectoryForFrame = function() {};
        var animationData = {interval: 80};
        var tick;
        var cleared = 0;
        var window = {
            setInterval: function(callback, ms) {
                if (ms !== 80) {throw Error('incorrect legacy interval');}
                tick = callback;
                return 7;
            },
            clearInterval: function(id) {
                if (id !== 7) {throw Error('wrong timer id');}
                cleared++;
            }
        };
    """
    scenario = """
        startPlayback();
        var results = [[currentFrame, isPlaying]];
        tick();
        results.push([currentFrame, isPlaying]);
        tick();
        results.push([currentFrame, isPlaying]);
        tick();                      // legacy mode loops to first sample
        results.push([currentFrame, isPlaying]);
        pausePlayback();
        results.push([currentFrame, isPlaying, cleared, playInterval]);
        console.log(JSON.stringify(results));
    """
    completed = subprocess.run(
        ["node", "-e", prelude + state_machine + scenario],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert json.loads(completed.stdout) == [
        [0, True],
        [1, True],
        [2, True],
        [0, True],
        [0, False, 1, None],
    ]
