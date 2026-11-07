"""Public T-04/B2 synchronized trapezoidal trajectory contract tests."""

from dataclasses import FrozenInstanceError, asdict, fields

import numpy as np
import pytest

from moro.trajectory import (
    JointTrajectory,
    PositionTrajectory,
    TrapezoidalProfileInfo,
    joint_trajectory,
    joint_waypoint_trajectory,
    joint_trapezoidal_trajectory,
    position_trajectory,
)


def test_synchronized_rr_trapezoid_matches_known_limits_and_phases():
    result = joint_trapezoidal_trajectory(
        [0.0, 0.0], [1.2, -0.8],
        v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        samples=27,
    )
    assert isinstance(result, JointTrajectory)
    assert result.method == "trapezoidal"
    assert result.samples == 27 and result.dof == 2
    assert result.q.shape == result.qd.shape == result.qdd.shape == (27, 2)
    assert result.profile.kind == "trapezoidal"
    assert result.profile.minimum_duration == pytest.approx(2.6)
    assert result.profile.duration == pytest.approx(2.6)
    assert result.profile.t_accel == pytest.approx(0.6)
    assert result.profile.t_cruise == pytest.approx(1.4)
    assert result.profile.t_decel == pytest.approx(0.6)
    assert result.profile.peak_path_speed == pytest.approx(0.5)
    assert result.profile.path_acceleration == pytest.approx(5 / 6)
    np.testing.assert_array_equal(result.q[0], [0.0, 0.0])
    np.testing.assert_array_equal(result.q[-1], [1.2, -0.8])
    np.testing.assert_array_equal(result.qd[[0, -1]], 0.0)
    np.testing.assert_allclose(result.q[:, 1], -(2 / 3) * result.q[:, 0])
    np.testing.assert_allclose(result.qd[:, 1], -(2 / 3) * result.qd[:, 0])
    np.testing.assert_allclose(result.qdd[:, 1], -(2 / 3) * result.qdd[:, 0])
    assert np.max(np.abs(result.qd[:, 0])) <= 0.6 + 1e-12
    assert np.max(np.abs(result.qd[:, 1])) <= 0.4 + 1e-12
    assert np.max(np.abs(result.qdd[:, 0])) <= 1.0 + 1e-12
    assert np.max(np.abs(result.qdd[:, 1])) <= 0.8 + 1e-12
    assert "method='trapezoidal'" in repr(result)


def test_profile_continuous_time_limits_from_analytic_extrema():
    q0 = np.array([0.0, 1.0, -3.0, 5.0])
    qf = np.array([1.2, -1.0, -3.0, 6.0])
    vmax = np.array([0.6, 0.7, 0.1, 0.5])
    amax = np.array([1.0, 0.8, 0.2, 0.3])
    delta = np.abs(qf - q0)
    result = joint_trapezoidal_trajectory(
        q0, qf, v_max=vmax, a_max=amax, samples=201
    )
    profile = result.profile
    # These are exact global maxima of the piecewise-constant/linear
    # acceleration/velocity profiles, not sample-only approximations.
    assert np.all(delta * profile.peak_path_speed <= vmax * (1 + 1e-12))
    assert np.all(delta * profile.path_acceleration <= amax * (1 + 1e-12))
    np.testing.assert_array_equal(result.q[:, 2], q0[2])
    np.testing.assert_array_equal(result.qd[:, 2], 0.0)
    np.testing.assert_array_equal(result.qdd[:, 2], 0.0)
    np.testing.assert_allclose(result.q[0], q0)
    np.testing.assert_allclose(result.q[-1], qf)
    assert np.all(np.diff(result.q[:, 0]) >= -1e-12)
    assert np.all(np.diff(result.q[:, 1]) <= 1e-12)


def test_requested_longer_duration_scales_velocity_and_acceleration():
    args = dict(
        q0=[1.0, -2.0], qf=[3.0, 1.0],
        v_max=[0.5, 0.75], a_max=[0.6, 0.9],
        samples=121,
    )
    base = joint_trapezoidal_trajectory(**args)
    slowed = joint_trapezoidal_trajectory(
        **args, duration=2 * base.duration
    )
    assert slowed.duration == pytest.approx(2 * base.duration)
    assert slowed.profile.minimum_duration == pytest.approx(base.duration)
    assert slowed.profile.t_accel == pytest.approx(
        2 * base.profile.t_accel
    )
    assert slowed.profile.t_cruise == pytest.approx(
        2 * base.profile.t_cruise
    )
    assert slowed.profile.peak_path_speed == pytest.approx(
        base.profile.peak_path_speed / 2
    )
    assert slowed.profile.path_acceleration == pytest.approx(
        base.profile.path_acceleration / 4
    )
    np.testing.assert_allclose(slowed.q, base.q, atol=1e-12)
    np.testing.assert_allclose(slowed.qd, base.qd / 2, atol=1e-12)
    np.testing.assert_allclose(slowed.qdd, base.qdd / 4, atol=1e-12)


def test_triangular_profile_single_dof_scalar_and_vector_inputs():
    a = joint_trapezoidal_trajectory(
        0.0, 1.0, v_max=2.0, a_max=1.0, samples=5
    )
    b = joint_trapezoidal_trajectory(
        [0.0], [1.0], v_max=[2.0], a_max=[1.0], samples=5
    )
    assert a.profile.kind == "triangular"
    assert a.profile.t_cruise == 0.0
    assert a.profile.duration == pytest.approx(2.0)
    np.testing.assert_array_equal(a.q[:, 0], [0.0, 0.125, 0.5, 0.875, 1.0])
    np.testing.assert_array_equal(a.t, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(a.q, b.q)
    np.testing.assert_allclose(a.qd, b.qd)
    np.testing.assert_allclose(a.qdd, b.qdd)


def test_sample_count_two_and_shifted_time_origin():
    result = joint_trapezoidal_trajectory(
        [0.0, 2.0], [1.0, -2.0],
        v_max=2.0, a_max=1.0, samples=2, t0=5.0,
    )
    assert result.t.shape == (2,)
    assert result.t[0] == 5.0
    assert result.t[-1] > 5.0
    np.testing.assert_array_equal(result.q[[0, 1]], [[0, 2], [1, -2]])
    np.testing.assert_array_equal(result.qd, 0.0)
    assert result.duration == pytest.approx(result.profile.duration)


def test_stationary_hold_with_explicit_duration():
    traj = joint_trapezoidal_trajectory(
        [0.3, -0.5], [0.3, -0.5],
        v_max=[0.6, 0.2], a_max=0.7,
        duration=3.0, t0=-2.0, samples=4,
    )
    assert traj.method == "trapezoidal"
    assert traj.profile.kind == "stationary"
    assert traj.profile.minimum_duration is None
    assert traj.profile.t_accel == traj.profile.t_decel == 0.0
    assert traj.profile.t_cruise == 3.0
    assert traj.profile.peak_path_speed == 0.0
    assert traj.profile.path_acceleration == 0.0
    np.testing.assert_allclose(traj.t, [-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(traj.q, np.tile([0.3, -0.5], (4, 1)))
    np.testing.assert_array_equal(traj.qd, 0.0)
    np.testing.assert_array_equal(traj.qdd, 0.0)


def test_stationary_without_duration_rejected():
    with pytest.raises(ValueError, match="stationary hold requires"):
        joint_trapezoidal_trajectory(0.0, 0.0, v_max=1.0, a_max=1.0)


def test_input_arrays_are_not_aliased():
    q0 = np.array([0.0, 1.0])
    qf = np.array([1.0, 0.0])
    v = np.array([1.0, 1.0])
    a = np.array([2.0, 2.0])
    result = joint_trapezoidal_trajectory(
        q0, qf, v_max=v, a_max=a, samples=7
    )
    saved = result.q.copy()
    q0[:] = 100.0
    qf[:] = 100.0
    v[:] = 100.0
    a[:] = 100.0
    np.testing.assert_array_equal(result.q, saved)


def test_joint_result_constructor_stays_backward_compatible():
    existing = JointTrajectory(
        [0.0, 1.0], [[0.0], [1.0]],
        [[0.0], [0.0]], [[0.0], [0.0]], "cubic",
    )
    assert existing.profile is None
    assert len(fields(JointTrajectory)) == 6
    assert [f.name for f in fields(JointTrajectory)] == [
        "t", "q", "qd", "qdd", "method", "profile",
    ]
    assert "profile" in asdict(existing)
    assert asdict(existing)["profile"] is None
    assert "profile" not in repr(existing)
    with pytest.raises(TypeError):
        JointTrajectory(
            [0.0, 1.0], [[0.0], [1.0]],
            [[0.0], [0.0]], [[0.0], [0.0]], "cubic", None,
        )
    assert "profile" not in [f.name for f in fields(PositionTrajectory)]


def test_trapezoidal_metadata_is_typed_immutable_and_serializable():
    traj = joint_trapezoidal_trajectory(
        0.0, 1.0, v_max=2.0, a_max=1.0
    )
    assert isinstance(traj.profile, TrapezoidalProfileInfo)
    assert asdict(traj)["profile"]["kind"] == "triangular"
    assert isinstance(traj.profile.duration, float)
    with pytest.raises(FrozenInstanceError):
        traj.profile.t_accel = 7
    with pytest.raises(ValueError, match="requires method"):
        JointTrajectory(
            traj.t, traj.q, traj.qd, traj.qdd, "cubic",
            profile=traj.profile,
        )
    with pytest.raises(ValueError, match="profile must"):
        JointTrajectory(
            traj.t, traj.q, traj.qd, traj.qdd, "trapezoidal",
            profile={"kind": "triangular"},
        )
    with pytest.raises(ValueError, match="profile.duration"):
        JointTrajectory(
            [0.0, 2.0], [[0.0], [1.0]],
            [[0.0], [0.0]], [[0.0], [0.0]], "trapezoidal",
            profile=traj.profile,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q0": [0.0, 1.0], "qf": [0.0]},
        {"q0": [0.0, np.inf], "qf": [1.0, 2.0]},
        {"q0": [0.0, 1.0], "qf": [1.0, 2 + 1j]},
        {"q0": [0.0, 1.0], "qf": [[1.0, 2.0]]},
        {"q0": [[0.0, 1.0]], "qf": [1.0, 2.0]},
        {"q0": 1e308, "qf": -1e308},
        {"q0": [0.0, 0.0], "qf": [1.0, 1.0], "v_max": [1.0]},
        {"q0": [0.0, 0.0], "qf": [1.0, 1.0], "a_max": [1.0, 2.0, 3.0]},
        {"v_max": False},
        {"v_max": 0.0},
        {"v_max": -1.0},
        {"v_max": np.nan},
        {"v_max": np.inf},
        {"v_max": [True]},
        {"v_max": [True, 1.0]},
        {"v_max": 1 + 2j},
        {"v_max": [[1.0]]},
        {"a_max": 0.0},
        {"a_max": np.nan},
        {"a_max": np.inf},
        {"a_max": True},
        {"a_max": [1.0, 0.0]},
        {"a_max": [1.0, np.nan]},
        {"duration": 0.0},
        {"duration": -1.0},
        {"duration": np.nan},
        {"duration": np.inf},
        {"duration": True},
        {"t0": np.nan},
        {"t0": np.inf},
        {"t0": True},
        {"t0": [0.0]},
        {"samples": 1},
        {"samples": True},
        {"samples": 2.1},
        {"samples": -2},
        {"samples": "4"},
    ],
)
def test_invalid_trapezoidal_inputs(kwargs):
    options = dict(q0=0.0, qf=1.0, v_max=1.0, a_max=1.0)
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        joint_trapezoidal_trajectory(**options)


def test_limit_vector_mixed_boolean_values_rejected():
    with pytest.raises(ValueError, match="v_max"):
        joint_trapezoidal_trajectory(
            [0.0, 0.0], [1.0, 1.0],
            v_max=[True, 1.0], a_max=[1.0, 1.0]
        )


def test_requested_shorter_than_minimum_is_rejected():
    with pytest.raises(ValueError, match="minimum feasible duration"):
        joint_trapezoidal_trajectory(
            0.0, 1.0, v_max=0.5, a_max=1.0, duration=0.5
        )


def test_time_grid_requires_distinguishable_floating_point_times():
    with pytest.raises(ValueError, match="Sampling requires"):
        joint_trapezoidal_trajectory(
            0.0, 1.0, v_max=1.0, a_max=1.0,
            t0=1e14, samples=10000,
        )
    with pytest.raises(ValueError, match="nonrepresentable endpoint"):
        joint_trapezoidal_trajectory(
            0.0, 1.0, v_max=1.0, a_max=1.0, t0=1e308,
        )


def test_polynomial_dispatch_remains_restricted():
    t = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="method"):
        joint_trajectory(0.0, 1.0, t, method="trapezoidal")
    with pytest.raises(ValueError, match="method"):
        position_trajectory(
            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
            t, method="trapezoidal"
        )
    with pytest.raises(ValueError, match="method"):
        joint_waypoint_trajectory(
            [0.0, 1.0], [0.0, 1.0], method="trapezoidal"
        )
