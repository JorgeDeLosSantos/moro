"""Multi-waypoint linear interpolation (0.7.0 PR-A2)."""

import numpy as np
import pytest

from moro.trajectory import (
    JointTrajectory,
    PositionTrajectory,
    joint_trajectory,
    joint_waypoint_trajectory,
    position_trajectory,
    position_waypoint_trajectory,
)


def test_joint_waypoints_and_time_major_shapes():
    waypoints = [[0.0, 1.0], [2.0, -1.0], [3.0, 0.0], [0.0, 2.0]]
    times = [5.0, 6.0, 8.0, 9.0]
    traj = joint_waypoint_trajectory(waypoints, times, samples_per_segment=3)
    assert isinstance(traj, JointTrajectory)
    assert traj.method == "linear"
    assert traj.dof == 2
    assert traj.samples == 7
    assert traj.duration == 4.0
    np.testing.assert_array_equal(traj.t, [5.0, 5.5, 6.0, 7.0, 8.0, 8.5, 9.0])
    np.testing.assert_allclose(traj.q[[0, 2, 4, 6]], waypoints)
    np.testing.assert_allclose(
        traj.qd[[0, 2, 4, 6]],
        [[2.0, -2.0], [0.5, 0.5], [-3.0, 2.0], [-3.0, 2.0]],
    )
    np.testing.assert_array_equal(traj.qdd, np.zeros((7, 2)))


def test_cartesian_waypoints_support_planar_path_with_z():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, -1.0, 0.0],
    ])
    result = position_waypoint_trajectory(
        points, [0.0, 1.0, 3.0], samples_per_segment=2
    )
    assert isinstance(result, PositionTrajectory)
    assert result.p.shape == result.v.shape == result.a.shape == (3, 3)
    np.testing.assert_allclose(result.p, points)
    np.testing.assert_allclose(result.v[1], [0.5, -1.5, 0.0])


def test_default_sampling_has_51_points_per_segment():
    traj = joint_waypoint_trajectory([0, 1, 2, 4], [0, 1, 2, 4])
    assert traj.t.shape == (151,)
    assert traj.q.shape == (151, 1)
    for knot in [0, 1, 2, 4]:
        assert np.count_nonzero(traj.t == knot) == 1


def test_one_dof_column_and_flat_inputs_agree():
    values = [0.0, 2.0, 3.0]
    t = [2.0, 5.0, 7.0]
    a = joint_waypoint_trajectory(values, t, samples_per_segment=5)
    b = joint_waypoint_trajectory(
        [[v] for v in values], t, samples_per_segment=5
    )
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_allclose(a.q, b.q)
    np.testing.assert_allclose(a.qd, b.qd)


def test_nonuniform_explicit_sample_times_are_preserved():
    samples = np.array([2.0, 2.1, 2.4, 3.5, 4.0, 5.8, 6.0])
    points = [0.0, 2.0, 5.0, 1.0]
    traj = joint_waypoint_trajectory(
        points, [2.0, 3.0, 4.5, 6.0], sample_times=samples
    )
    np.testing.assert_array_equal(traj.t, samples)
    assert 3.0 not in traj.t
    assert 4.5 not in traj.t
    np.testing.assert_allclose(
        traj.q[:, 0],
        np.interp(samples, [2.0, 3.0, 4.5, 6.0], points),
    )


def test_trajectory_is_invariant_under_time_shift():
    values = [[0.0, 0.5], [1.0, -1.0], [2.0, 3.0]]
    a = joint_waypoint_trajectory(values, [0, 1, 3], samples_per_segment=9)
    b = joint_waypoint_trajectory(values, [12, 13, 15], samples_per_segment=9)
    np.testing.assert_allclose(b.t, a.t + 12)
    np.testing.assert_allclose(b.q, a.q)
    np.testing.assert_allclose(b.qd, a.qd)
    np.testing.assert_allclose(b.qdd, a.qdd)


def test_linear_two_waypoint_equivalence_joint_and_position():
    t = [1.0, 1.25, 2.0, 3.0]
    old_joint = joint_trajectory([1.0, -2.0], [3.0, 0.0], t, method="linear")
    new_joint = joint_waypoint_trajectory(
        [[1.0, -2.0], [3.0, 0.0]], [1.0, 3.0],
        method="linear", sample_times=t,
    )
    for a, b in zip(
        (new_joint.t, new_joint.q, new_joint.qd, new_joint.qdd),
        (old_joint.t, old_joint.q, old_joint.qd, old_joint.qdd),
    ):
        np.testing.assert_allclose(a, b)

    old_cart = position_trajectory([0, 0, 0], [1, 2, 3], t, method="linear")
    new_cart = position_waypoint_trajectory(
        [[0, 0, 0], [1, 2, 3]], [1.0, 3.0], sample_times=t
    )
    for a, b in zip(
        (new_cart.t, new_cart.p, new_cart.v, new_cart.a),
        (old_cart.t, old_cart.p, old_cart.v, old_cart.a),
    ):
        np.testing.assert_allclose(a, b)


def test_linear_corner_uses_right_hand_velocity_and_final_left_hand():
    traj = joint_waypoint_trajectory(
        [0.0, 1.0, -1.0], [0.0, 1.0, 3.0],
        sample_times=[0.0, 0.5, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(traj.qd[:, 0], [1.0, 1.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(traj.qdd, np.zeros((5, 1)))


def test_repeated_positions_and_stationary_segments_are_allowed():
    traj = joint_waypoint_trajectory(
        [0, 1, 1, 2], [0, 1, 2, 3], samples_per_segment=3
    )
    np.testing.assert_allclose(traj.q[2:5, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(traj.qd[2:4, 0], [0.0, 0.0])


def test_samples_per_segment_two_has_every_knot_once():
    result = joint_waypoint_trajectory(
        [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0],
        samples_per_segment=2,
    )
    np.testing.assert_array_equal(result.t, [0.0, 1.0, 2.0, 3.0])


def test_method_and_policy_are_case_insensitive():
    traj = joint_waypoint_trajectory(
        [0, 1], [0, 2], method="LINEAR", derivative_policy="ZERO"
    )
    assert traj.method == "linear"


@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [1.0],
        [[1.0, 2.0]],
        [[], []],
        [[0.0], [np.nan]],
        [[0.0], [np.inf]],
        [[0.0], [1 + 2j]],
        [[0.0, 1.0], [2.0]],
        np.zeros((2, 1, 1)),
        [["a"], ["b"]],
    ],
)
def test_invalid_joint_waypoints(waypoints):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory(waypoints, [0, 1])


@pytest.mark.parametrize(
    "waypoints",
    [
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0], [1.0]],
        [0.0, 1.0, 2.0],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0]],
        np.zeros((2, 3, 1)),
    ],
)
def test_invalid_cartesian_waypoints(waypoints):
    with pytest.raises(ValueError):
        position_waypoint_trajectory(waypoints, [0.0, 1.0])


@pytest.mark.parametrize(
    "times",
    [
        [0.0],
        [0, 0],
        [2, 1],
        [0, np.nan],
        [0, np.inf],
        [[0, 1]],
        [0, 1, 2],
        [1e308, -1e308],
    ],
)
def test_invalid_waypoint_times(times):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory([0, 1], times)


@pytest.mark.parametrize(
    "sample_times",
    [
        [],
        [0],
        [0.1, 1.0],
        [0.0, 0.9],
        [0.0, 1.1],
        [-0.1, 1.0],
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.8, 0.7, 1.0],
        [0.0, np.nan, 1.0],
        [0.0, np.inf],
        [[0.0, 1.0]],
    ],
)
def test_invalid_sample_times(sample_times):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory([0, 1], [0, 1], sample_times=sample_times)


@pytest.mark.parametrize("count", [True, False, 1, 0, -2, 2.1, 4.0, np.nan, "3"])
def test_invalid_samples_per_segment(count):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory([0, 1], [0, 1], samples_per_segment=count)


def test_sampling_controls_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot both"):
        joint_waypoint_trajectory(
            [0, 1], [0, 1], sample_times=[0, 1], samples_per_segment=3
        )


@pytest.mark.parametrize(
    "method,policy,velocities,accelerations",
    [
        ("cubic", "zero", None, None),
        ("quintic", "zero", None, None),
        ("invalid", "zero", None, None),
        ("linear", "shape_preserving", None, None),
        ("linear", 2, None, None),
        ("linear", "zero", [0, 0], None),
        ("linear", "zero", None, [0, 0]),
    ],
)
def test_unsupported_methods_and_derivatives_are_rejected(
    method, policy, velocities, accelerations
):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory(
            [0, 1], [0, 1], method=method, derivative_policy=policy,
            velocities=velocities, accelerations=accelerations,
        )


def test_outputs_do_not_alias_user_inputs():
    points = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    waypoint_times = np.array([0.0, 1.0, 2.0])
    samples = np.array([0.0, 0.25, 1.25, 2.0])
    traj = joint_waypoint_trajectory(
        points, waypoint_times, sample_times=samples
    )
    saved_q = traj.q.copy()
    saved_t = traj.t.copy()
    points[:] = 99
    waypoint_times[:] = 99
    samples[:] = 99
    np.testing.assert_array_equal(traj.q, saved_q)
    np.testing.assert_array_equal(traj.t, saved_t)


def test_overflowing_segment_slopes_are_rejected():
    with pytest.raises(ValueError, match="velocities must be finite"):
        joint_waypoint_trajectory(
            [-1e308, 1e308], [0.0, 1.0], samples_per_segment=2
        )
