"""S3 acceptance: combine trajectory, dynamics, playback and workspace APIs.

Browser interaction is checked separately: CI asserts mathematical results,
return contracts, and serializable HTML instead of claiming WebGL execution.
"""

import json
import re

import numpy as np
import pytest
import sympy as sp
from IPython.display import HTML

from moro import Robot
from moro.abc import q1, q2
from moro.dynamics import (
    forward_dynamics, inverse_dynamics_trajectory,
)
from moro.trajectory import (
    joint_waypoint_trajectory, joint_trapezoidal_trajectory,
)
from moro.visualization import (
    RobotVisualizer, plot_workspace, plot_workspace_section,
)
from moro.workspace import sample_workspace, section_workspace


def _robot():
    robot = Robot(
        (0.9, 0, 0, q1, "r"),
        (0.6, 0, 0, q2, "r"),
    )
    robot.masses = [1.5, 1.0]
    robot.cm_positions = [(-0.45, 0, 0), (-0.3, 0, 0)]
    robot.inertia_tensors = [
        sp.diag(0, 0, 0.06),
        sp.diag(0, 0, 0.025),
    ]
    robot.gravity = (0, -9.81, 0)
    return robot


def _workspace_payload(viewer):
    assert isinstance(viewer, HTML)
    match = re.search(r"var data = (\{.*?\});\s*var framing", viewer.data, re.S)
    assert match, "Three.js HTML did not contain the serialized point cloud"
    return json.loads(match.group(1))


def test_nonuniform_waypoints_to_threejs_playback_and_inverse_dynamics():
    robot = _robot()
    q_samples = np.array([[0.0, 0.0], [0.35, -0.2], [0.6, 0.1]])
    times = np.array([1.0, 1.4, 2.3])
    sample_times = np.array([1.0, 1.05, 1.2, 1.4, 1.6, 2.0, 2.3])
    motion = joint_waypoint_trajectory(
        q_samples, times, method="quintic",
        derivative_policy="zero", sample_times=sample_times,
    )
    assert motion.q.shape == motion.qd.shape == motion.qdd.shape == (7, 2)
    np.testing.assert_array_equal(motion.t, sample_times)
    np.testing.assert_allclose(motion.q[[0, 3, -1]], q_samples)
    np.testing.assert_allclose(motion.qd[[0, 3, -1]], 0, atol=1e-10)

    viewer = RobotVisualizer(robot).animate(
        motion.q, times=motion.t, playback_speed=1.5,
    )
    assert isinstance(viewer, HTML)
    animation = re.search(r"var animationData = (\{[^\n]+\});", viewer.data)
    assert animation is not None
    payload = json.loads(animation.group(1))
    np.testing.assert_allclose(
        payload["relative_times"], motion.t - motion.t[0]
    )
    assert len(payload["frames"]) == motion.samples
    assert payload["playback_speed"] == 1.5

    tau = inverse_dynamics_trajectory(robot, motion)
    assert tau.shape == motion.q.shape
    assert np.isfinite(tau).all()
    for index in [0, 3, -1]:
        qdd = forward_dynamics(
            robot, motion.q[index], motion.qd[index], tau[index]
        )
        np.testing.assert_allclose(qdd, motion.qdd[index], atol=1e-9, rtol=1e-9)


def test_trapezoidal_motion_to_dynamics_and_playback():
    robot = _robot()
    motion = joint_trapezoidal_trajectory(
        [0.0, 0.0], [0.5, -0.3],
        v_max=[0.5, 0.4], a_max=[1.0, 0.9], samples=17,
    )
    tau = inverse_dynamics_trajectory(robot, motion)
    assert motion.samples == 17
    assert tau.shape == motion.q.shape == (17, 2)
    assert np.isfinite(tau).all()
    viewer = RobotVisualizer(robot).animate(motion.q, times=motion.t)
    assert isinstance(viewer, HTML)


def test_workspace_sampling_sections_and_two_backends():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.0, 0, q2, 0, "p"),
    )
    robot.joint_limits = [(-np.pi, np.pi), (0.0, 0.5)]
    workspace = sample_workspace(
        robot, method="grid", grid_shape=(12, 5),
    )
    assert workspace.configurations.shape == (60, 2)
    assert workspace.points.shape == (60, 3)

    section = section_workspace(
        workspace, plane="xy", level=0.25, thickness=0.0,
    )
    assert section.samples == 12
    np.testing.assert_array_equal(
        section.points, workspace.points[section.indices]
    )
    np.testing.assert_array_equal(
        section.configurations, workspace.configurations[section.indices]
    )
    original = workspace.points.copy()
    cloud = _workspace_payload(plot_workspace(workspace, show_grid=True))
    plane = _workspace_payload(
        plot_workspace_section(section, show_grid=True)
    )
    assert cloud["projection"] == "3d"
    assert cloud["show_grid"] is True
    assert plane["projection"] == "xy"
    assert plane["show_grid"] is True
    assert plane["section"]["source_samples"] == workspace.samples
    assert len(plane["positions"]) == section.samples
    np.testing.assert_allclose(
        np.asarray(plane["positions"])[:, 2], 0.25,
    )
    np.testing.assert_array_equal(workspace.points, original)

    fig, ax = plot_workspace_section(section, backend="matplotlib")
    try:
        assert ax.figure is fig
        assert len(ax.collections[0].get_offsets()) == section.samples
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_empty_section_works_across_both_backends():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.0, 0, q2, 0, "p"),
    )
    robot.joint_limits = [(-np.pi, np.pi), (0.0, 0.5)]
    workspace = sample_workspace(robot, method="grid", grid_shape=(8, 3))
    empty = section_workspace(
        workspace, plane="xy", level=4.0, thickness=0,
    )
    assert empty.samples == 0
    html = plot_workspace_section(empty, show_grid=True)
    payload = _workspace_payload(html)
    assert payload["positions"] == []
    assert payload["framing"]["center"][2] == 4.0
    fig, ax = plot_workspace_section(empty, backend="matplotlib")
    try:
        assert np.asarray(ax.collections[0].get_offsets()).shape == (0, 2)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
