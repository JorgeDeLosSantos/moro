"""Cubic/quintic piecewise interpolation tests for Moro 0.7.0 PR-A3."""

import numpy as np
import pytest

from moro.trajectory import (
    _evaluate_polynomial_segment,
    joint_trajectory,
    joint_waypoint_trajectory,
    position_trajectory,
    position_waypoint_trajectory,
)


@pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
def test_two_waypoint_equivalence_joint_with_nonzero_boundaries(method):
    sample_times = [2.0, 2.15, 2.6, 4.0]
    points = [[0.0, 0.5], [1.0, -2.0]]
    kwargs = {}
    old_kwargs = {}
    if method != "linear":
        velocities = [[0.3, -0.2], [0.8, 0.4]]
        kwargs["velocities"] = velocities
        old_kwargs.update(qd0=velocities[0], qdf=velocities[1])
    if method == "quintic":
        accelerations = [[0.7, 0.2], [-0.3, 0.1]]
        kwargs["accelerations"] = accelerations
        old_kwargs.update(qdd0=accelerations[0], qddf=accelerations[1])
    actual = joint_waypoint_trajectory(
        points, [2.0, 4.0], method=method,
        sample_times=sample_times, **kwargs
    )
    expected = joint_trajectory(
        points[0], points[1], sample_times,
        method=method, **old_kwargs
    )
    for a, b in zip(
        (actual.t, actual.q, actual.qd, actual.qdd),
        (expected.t, expected.q, expected.qd, expected.qdd),
    ):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["cubic", "quintic"])
def test_cartesian_two_waypoint_equivalence(method):
    points = [[0.0, 1.0, -0.4], [2.0, -1.0, 1.2]]
    times = [0.0, 0.2, 0.8, 1.0]
    velocities = [[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]]
    kwargs = {"velocities": velocities}
    old_kwargs = {"v0": velocities[0], "vf": velocities[1]}
    if method == "quintic":
        accelerations = [[0.2, -0.3, 0.1], [-0.1, 0.2, 0.4]]
        kwargs["accelerations"] = accelerations
        old_kwargs.update(a0=accelerations[0], af=accelerations[1])
    actual = position_waypoint_trajectory(
        points, [0.0, 1.0], sample_times=times, method=method, **kwargs
    )
    expected = position_trajectory(
        points[0], points[1], times, method=method, **old_kwargs
    )
    for a, b in zip(
        (actual.t, actual.p, actual.v, actual.a),
        (expected.t, expected.p, expected.v, expected.a),
    ):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["cubic", "quintic"])
def test_continuity_from_analytic_segment_boundaries(method):
    points = np.array([[0.0, 0.5], [1.0, 2.0], [0.3, 1.2], [2.1, 0.2]])
    times = np.array([2.0, 2.9, 5.0, 7.0])
    v = np.array([[0.0, 0.0], [0.4, -0.2], [0.1, 0.3], [0.0, 0.0]])
    a = np.array([[0.0, 0.0], [0.1, -0.3], [0.4, 0.2], [0.0, 0.0]])
    args = {"velocities": v}
    if method == "quintic":
        args["accelerations"] = a
    result = joint_waypoint_trajectory(
        points, times, method=method, samples_per_segment=5, **args
    )
    for i in range(1, len(times) - 1):
        left_kwargs = {"v0": v[i-1], "vf": v[i]}
        right_kwargs = {"v0": v[i], "vf": v[i+1]}
        if method == "quintic":
            left_kwargs.update(a0=a[i-1], af=a[i])
            right_kwargs.update(a0=a[i], af=a[i+1])
        left = _evaluate_polynomial_segment(
            points[i-1], points[i], times[i-1], times[i],
            [times[i]], method=method, **left_kwargs,
        )
        right = _evaluate_polynomial_segment(
            points[i], points[i+1], times[i], times[i+1],
            [times[i]], method=method, **right_kwargs,
        )
        for k in ((1, 2, 3) if method == "quintic" else (1, 2)):
            np.testing.assert_allclose(
                left[k], right[k], atol=1e-12, rtol=1e-12
            )
        knot_idx = np.flatnonzero(result.t == times[i])
        assert knot_idx.size == 1
        np.testing.assert_allclose(result.q[knot_idx], points[i:i+1])
        np.testing.assert_allclose(result.qd[knot_idx], v[i:i+1])
        if method == "quintic":
            np.testing.assert_allclose(result.qdd[knot_idx], a[i:i+1])


@pytest.mark.parametrize("method", ["cubic", "quintic"])
def test_default_zero_derivatives_stop_at_each_waypoint(method):
    waypoints = [0.0, 0.6, -0.2, 1.0]
    times = [0.0, 1.0, 2.5, 4.0]
    traj = joint_waypoint_trajectory(
        waypoints, times, method=method, samples_per_segment=4
    )
    for t in times:
        i = np.flatnonzero(traj.t == t)
        assert i.size == 1
        np.testing.assert_allclose(traj.qd[i], 0.0, atol=1e-12)
        if method == "quintic":
            np.testing.assert_allclose(traj.qdd[i], 0.0, atol=1e-12)


def test_quintic_is_default_waypoint_method():
    actual = joint_waypoint_trajectory([0.0, 1.0, 0.5], [0.0, 1.0, 3.0])
    expected = joint_waypoint_trajectory(
        [0.0, 1.0, 0.5], [0.0, 1.0, 3.0], method="quintic"
    )
    assert actual.method == "quintic"
    np.testing.assert_allclose(actual.q, expected.q)
    assert position_waypoint_trajectory(
        [[0, 0, 0], [1, 0, 0]], [0.0, 1.0]
    ).method == "quintic"


def test_sparse_samples_can_omit_interior_knots():
    times = [1.0, 3.0, 6.0]
    grid = [1.0, 1.2, 2.0, 4.0, 6.0]
    result = joint_waypoint_trajectory(
        [0, 1, 2], times, sample_times=grid, method="quintic"
    )
    np.testing.assert_array_equal(result.t, grid)
    for i, point in enumerate([0, 1, 2]):
        if i == 1:
            continue
        np.testing.assert_allclose(result.q[0 if i == 0 else -1, 0], point)


def test_1dof_derivative_vector_and_column_are_equivalent():
    vals = [0.0, 1.0, -0.5]
    times = [0.0, 1.0, 3.0]
    u = joint_waypoint_trajectory(
        vals, times, method="quintic",
        velocities=[0.0, 0.2, 0.0],
        accelerations=[0.0, 0.1, 0.0],
    )
    v = joint_waypoint_trajectory(
        vals, times, method="quintic",
        velocities=[[0.0], [0.2], [0.0]],
        accelerations=[[0.0], [0.1], [0.0]],
    )
    np.testing.assert_allclose(u.q, v.q)
    np.testing.assert_allclose(u.qd, v.qd)
    np.testing.assert_allclose(u.qdd, v.qdd)


def test_quintic_velocities_prescribed_accelerations_default_zero():
    q = joint_waypoint_trajectory(
        [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
        velocities=[0.0, 0.4, 0.0], method="quintic",
        samples_per_segment=3,
    )
    np.testing.assert_allclose(
        q.qd[[0, 2, 4], 0], [0.0, 0.4, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(q.qdd[[0, 2, 4], 0], 0.0, atol=1e-12)


@pytest.mark.parametrize("method", ["cubic", "quintic"])
def test_derivative_time_scaling(method):
    vals = [[0.0, 1.0], [2.0, -2.0], [0.5, 1.2]]
    times = [0.0, 1.0, 3.0]
    velocities = np.array([[0.1, 0.2], [0.3, -0.1], [0.0, 0.0]])
    kwargs = {"velocities": velocities}
    if method == "quintic":
        kwargs["accelerations"] = np.array(
            [[0.0, 0.1], [-0.2, 0.1], [0.0, 0.0]]
        )
    orig = joint_waypoint_trajectory(
        vals, times, method=method, samples_per_segment=7, **kwargs
    )
    scaled = joint_waypoint_trajectory(
        vals, [2*t+5 for t in times], method=method,
        samples_per_segment=7, velocities=velocities / 2.0,
        **({"accelerations": kwargs["accelerations"] / 4.0}
           if method == "quintic" else {}),
    )
    np.testing.assert_allclose(scaled.q, orig.q)
    np.testing.assert_allclose(scaled.qd, orig.qd / 2.0)
    np.testing.assert_allclose(scaled.qdd, orig.qdd / 4.0)


@pytest.mark.parametrize(
    "method,policy,v,a",
    [
        ("linear", "zero", [0, 0], None),
        ("linear", "zero", None, [0, 0]),
        ("cubic", "zero", None, [0, 0]),
        ("quintic", "shape_preserving", None, None),
        ("linear", "shape_preserving", None, None),
        ("cubic", "unknown", None, None),
    ],
)
def test_incompatible_policy_and_boundary_conditions(method, policy, v, a):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory(
            [0, 1], [0, 1], method=method, derivative_policy=policy,
            velocities=v, accelerations=a,
        )


@pytest.mark.parametrize(
    "bad",
    [
        0.5,
        [0.0, 0.1],
        [0.0, 0.1, 0.2, 0.3],
        [[0.0, 0.1], [0.1, 0.0]],
        [[0.0], [np.inf], [0.0]],
        [[0.0], [np.nan], [0.0]],
        np.zeros((3, 1, 1)),
    ],
)
def test_invalid_one_dof_derivative_shapes(bad):
    with pytest.raises(ValueError):
        joint_waypoint_trajectory(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            method="cubic", velocities=bad,
        )


def test_cartesian_derivative_requires_k_by_3():
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]]
    with pytest.raises(ValueError, match="velocities"):
        position_waypoint_trajectory(
            points, [0.0, 1.0], method="cubic", velocities=[0.0, 0.0]
        )
    with pytest.raises(ValueError, match="accelerations"):
        position_waypoint_trajectory(
            points, [0.0, 1.0], method="quintic", accelerations=[[0, 0], [0, 0]]
        )


def test_nonzero_derivatives_with_repeated_positions_are_allowed():
    result = joint_waypoint_trajectory(
        [0.0, 0.0, 1.0], [0.0, 1.0, 2.0],
        method="cubic", velocities=[0.1, -0.1, 0.0],
        samples_per_segment=7,
    )
    np.testing.assert_allclose(result.q[[0, 6, 12], 0], [0, 0, 1])
    assert np.max(np.abs(result.q[:7, 0])) > 0.0
