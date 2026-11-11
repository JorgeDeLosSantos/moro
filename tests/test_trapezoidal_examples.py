"""Regression checks for Moro 0.7.0 T-04/B3 teaching examples.

These cases exercise the public trajectory results and the same numerical
setups shown in the R and RR notebook-style documentation examples.
"""

import numpy as np
import pytest

from moro.trajectory import (
    JointTrajectory,
    TrapezoidalProfileInfo,
    joint_trajectory,
    joint_trapezoidal_trajectory,
)


def test_r_example_matches_reported_three_phase_schedule():
    motion = joint_trapezoidal_trajectory(
        -0.3, 0.9, v_max=0.6, a_max=1.0, samples=261,
    )
    info = motion.profile

    assert isinstance(motion, JointTrajectory)
    assert info.kind == "trapezoidal"
    assert motion.duration == pytest.approx(2.6)
    assert info.duration == pytest.approx(motion.duration)
    assert info.minimum_duration == pytest.approx(2.6)
    assert info.t_accel == pytest.approx(0.6)
    assert info.t_cruise == pytest.approx(1.4)
    assert info.t_decel == pytest.approx(0.6)
    assert motion.q.shape == motion.qd.shape == motion.qdd.shape == (261, 1)
    assert motion.t.shape == (261,)
    np.testing.assert_allclose(motion.q[[0, -1], 0], [-0.3, 0.9])
    np.testing.assert_array_equal(motion.qd[[0, -1], 0], 0.0)
    assert np.max(np.abs(motion.qd)) <= 0.6 + 1e-12
    assert np.max(np.abs(motion.qdd)) <= 1.0 + 1e-12


def test_rr_example_is_synchronized_and_uses_physical_joint_limits():
    q0 = np.array([0.0, 0.0])
    qf = np.array([1.2, -0.8])
    v_max = np.array([0.6, 0.4])
    a_max = np.array([1.0, 0.8])
    rr = joint_trapezoidal_trajectory(
        q0, qf, v_max=v_max, a_max=a_max, samples=261,
    )

    np.testing.assert_allclose(rr.q[:, 1], -(2 / 3) * rr.q[:, 0], atol=1e-12)
    np.testing.assert_allclose(rr.qd[:, 1], -(2 / 3) * rr.qd[:, 0], atol=1e-12)
    np.testing.assert_allclose(rr.qdd[:, 1], -(2 / 3) * rr.qdd[:, 0], atol=1e-12)
    info = rr.profile
    delta = np.abs(qf - q0)
    np.testing.assert_array_less(
        delta * info.peak_path_speed, v_max + 1e-12,
    )
    np.testing.assert_array_less(
        delta * info.path_acceleration, a_max + 1e-12,
    )


def test_equal_duration_quintic_comparison_preserves_endpoints_but_not_boundary_acceleration():
    q0 = [0.0, 0.0]
    qf = [1.2, -0.8]
    trap = joint_trapezoidal_trajectory(
        q0, qf, v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        samples=261,
    )
    quintic = joint_trajectory(q0, qf, trap.t, method="quintic")

    np.testing.assert_array_equal(trap.t, quintic.t)
    np.testing.assert_allclose(trap.q[[0, -1]], quintic.q[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(trap.qd[[0, -1]], 0.0, atol=1e-12)
    np.testing.assert_allclose(quintic.qd[[0, -1]], 0.0, atol=1e-12)
    np.testing.assert_allclose(quintic.qdd[[0, -1]], 0.0, atol=1e-12)
    assert abs(trap.qdd[0, 0]) > 0.1
    assert abs(trap.qdd[-1, 0]) > 0.1
    assert quintic.profile is None
    assert trap.profile.kind == "trapezoidal"


def test_duration_scaling_in_rr_example_does_not_change_configuration_path():
    args = dict(
        q0=[0.0, 0.0], qf=[1.2, -0.8],
        v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        samples=261,
    )
    fast = joint_trapezoidal_trajectory(**args)
    slow = joint_trapezoidal_trajectory(
        **args, duration=2 * fast.duration,
    )
    np.testing.assert_allclose(slow.q, fast.q, atol=1e-12)
    np.testing.assert_allclose(slow.qd, fast.qd / 2, atol=1e-12)
    np.testing.assert_allclose(slow.qdd, fast.qdd / 4, atol=1e-12)
    assert slow.profile.duration == pytest.approx(2 * fast.profile.duration)


@pytest.mark.parametrize(
    "q0,qf,v_max,a_max",
    [
        ([0.0, 0.0], [1.0, 0.25], [10.0, 0.25], [1.0, 10.0]),
        ([0.0, 1.0], [-0.7, 1.0], [0.4, 1e-6], [0.9, 1e-6]),
        ([0.2, -1.0, 0.0], [-0.5, 0.0, 0.25],
         [0.4, 0.2, 0.1], [0.7, 0.3, 0.8]),
        ([0.0], [-1.0], [2.0], [0.5]),
    ],
)
def test_analytic_global_bounds_for_different_limiting_joints(
    q0, qf, v_max, a_max,
):
    traj = joint_trapezoidal_trajectory(
        q0, qf, v_max=v_max, a_max=a_max, samples=47,
    )
    displacement = np.abs(np.asarray(qf) - np.asarray(q0))
    info = traj.profile
    # The maximum of a piecewise-linear speed occurs exactly at the peak;
    # piecewise-constant acceleration has its exact maximum in each phase.
    assert np.all(
        displacement * info.peak_path_speed <= np.asarray(v_max) + 1e-12
    )
    assert np.all(
        displacement * info.path_acceleration <= np.asarray(a_max) + 1e-12
    )
    np.testing.assert_allclose(traj.q[0], q0)
    np.testing.assert_allclose(traj.q[-1], qf)
    np.testing.assert_allclose(traj.qd[[0, -1]], 0.0, atol=1e-12)
    assert np.all(np.diff(traj.t) > 0)


def test_zero_displacement_coordinates_do_not_constrain_progress():
    regular = joint_trapezoidal_trajectory(
        0.0, 1.0, v_max=0.5, a_max=1.0,
    )
    with_stationary_joint = joint_trapezoidal_trajectory(
        [0.0, 2.0], [1.0, 2.0],
        v_max=[0.5, 1e-20], a_max=[1.0, 1e-20],
    )
    assert with_stationary_joint.duration == pytest.approx(regular.duration)
    np.testing.assert_allclose(with_stationary_joint.q[:, 0],
                               regular.q[:, 0], atol=1e-12)
    np.testing.assert_array_equal(with_stationary_joint.q[:, 1], 2.0)
    np.testing.assert_array_equal(with_stationary_joint.qd[:, 1], 0.0)
    np.testing.assert_array_equal(with_stationary_joint.qdd[:, 1], 0.0)


def test_shifting_time_origin_does_not_change_motion():
    args = dict(
        q0=[0.0, -0.5], qf=[1.2, 0.3],
        v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        samples=111,
    )
    origin = joint_trapezoidal_trajectory(**args)
    shifted = joint_trapezoidal_trajectory(**args, t0=5.0)
    np.testing.assert_allclose(shifted.t - 5.0, origin.t, atol=1e-12)
    np.testing.assert_allclose(shifted.q, origin.q, atol=1e-12)
    np.testing.assert_allclose(shifted.qd, origin.qd, atol=1e-12)
    np.testing.assert_allclose(shifted.qdd, origin.qdd, atol=1e-12)


@pytest.mark.parametrize(
    "kind,t_accel,t_cruise,t_decel,minimum_duration,peak_speed,accel",
    [
        ("unknown", 1.0, 0.0, 1.0, 2.0, 1.0, 1.0),
        ("triangular", 1.0, 0.1, 0.9, 2.0, 1.0, 1.0),
        ("trapezoidal", 1.0, 0.0, 1.0, 2.0, 1.0, 1.0),
        ("triangular", 1.0, 0.0, 0.5, 2.0, 1.0, 1.0),
        ("triangular", 1.0, 0.0, 1.0, None, 1.0, 1.0),
        ("stationary", 0.0, 2.0, 0.0, 2.0, 0.0, 0.0),
        ("stationary", 1.0, 0.0, 1.0, None, 0.0, 0.0),
    ],
)
def test_profile_info_rejects_invalid_phase_metadata(
    kind,t_accel,t_cruise,t_decel,minimum_duration,peak_speed,accel,
):
    with pytest.raises(ValueError):
        TrapezoidalProfileInfo(
            kind=kind,
            t_accel=t_accel,
            t_cruise=t_cruise,
            t_decel=t_decel,
            duration=2.0,
            minimum_duration=minimum_duration,
            peak_path_speed=peak_speed,
            path_acceleration=accel,
        )


def test_hold_keeps_metadata_unambiguous():
    hold = joint_trapezoidal_trajectory(
        [0.2, -0.1], [0.2, -0.1],
        v_max=[0.6, 0.4], a_max=[1.0, 0.8],
        duration=2.0,
    )
    assert hold.profile.kind == "stationary"
    assert hold.profile.t_cruise == hold.duration
    assert hold.profile.minimum_duration is None
    np.testing.assert_array_equal(hold.qd, 0.0)
    np.testing.assert_array_equal(hold.qdd, 0.0)
