"""Regression witnesses for the Moro 0.7.1 IK robustness investigation.

Two strict xfails document reproducible *current* LM failures. Remove their
xfail markers only when the solver fix makes the default calls converge.
The other tests establish known-reachable controls without requiring
the solver to return a particular redundant/periodic joint configuration.
"""

import numpy as np
import pytest
import sympy as sp

from moro.abc import q1, q2, q3
from moro.core import Robot
from moro.inverse_kinematics import solve_position_ik


@pytest.fixture(params=["rr", "rrr"])
def regression_case(request):
    if request.param == "rr":
        return (
            lambda scale: Robot((200 * scale, 0, 0, q1),
                                (200 * scale, 0, 0, q2)),
            np.array([200.0, 200.0, 0.0]),
            [0.1, 0.1],
        )
    return (
        lambda scale: Robot((0, sp.pi / 2, 200 * scale, q1),
                            (200 * scale, 0, 0, q2),
                            (200 * scale, 0, 0, q3)),
        np.array([200.0, 200.0, 200.0]),
        [0.5, 0.5, 0.5],
    )


def test_lm_default_does_not_terminate_before_first_accepted_step(regression_case):
    make_robot, target, q0 = regression_case
    solution = solve_position_ik(make_robot(1.0), target, q0=q0)

    # Regression: default LM must adapt damping past initial rejected trials.
    assert solution.converged
    assert solution.error < 1e-6


def test_relaxed_stagnation_recovers_reported_reachable_target(regression_case):
    make_robot, target, q0 = regression_case
    robot = make_robot(1.0)
    solution = solve_position_ik(
        robot, target, q0=q0, stagnation_iterations=6,
    )
    assert solution.converged
    actual = np.asarray(
        robot.T[:3, 3].subs(dict(zip(robot.qs, solution.q))), dtype=float
    ).reshape(3)
    np.testing.assert_allclose(actual, target, atol=1e-6, rtol=0)


def test_physically_equivalent_m_and_mm_targets_are_reachable(regression_case):
    make_robot, target_mm, q0 = regression_case
    # Compare identical physical precision: 1e-6 mm = 1e-9 m.
    # error_change_tol must also scale with length; joint step_tol is angular.
    for scale in (1.0, 1e-3):
        solution = solve_position_ik(
            make_robot(scale),
            target_mm * scale,
            q0=q0,
            tol=1e-6 * scale,
            error_change_tol=1e-12 * scale,
            stagnation_iterations=6,
        )
        assert solution.converged
        assert solution.error < 1e-6 * scale


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_fixed_prismatic_joint_stays_infeasible_and_terminates(scale):
    """A constrained target must not become a false success after LM hardening."""
    robot = Robot((0, 0, q1, 0, "p"))
    solution = solve_position_ik(
        robot,
        [0, 0, 3 * scale],
        q0=[0],
        joint_limits=[(0, 1 * scale)],
        tol=1e-6 * scale,
        error_change_tol=1e-12 * scale,
        stagnation_iterations=6,
        max_iter=100,
    )
    assert not solution.converged
    assert 0 <= solution.q[0] <= scale
    assert solution.iterations <= 100
    assert solution.error >= 2 * scale - 1e-9 * scale


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_mixed_revolute_prismatic_reachable_with_explicit_scaled_limits(scale):
    """One R/P example exercises angular versus translational Jacobian columns."""
    robot = Robot((200 * scale, 0, 0, q1),
                  (0, 0, q2, 0, "p"))
    target = np.asarray(
        robot.T[:3, 3].subs({q1: 0.5, q2: 80 * scale}), dtype=float
    ).reshape(3)
    solution = solve_position_ik(
        robot,
        target,
        q0=[0.49, 70 * scale],
        joint_limits=[(-np.pi, np.pi), (0, 100 * scale)],
        tol=1e-6 * scale,
        error_change_tol=1e-12 * scale,
        stagnation_iterations=12,
        max_iter=200,
    )
    assert solution.converged
    assert abs(solution.error) < 1e-6 * scale
    assert 0 <= solution.q[1] <= 100 * scale


def test_seeded_random_start_reproducible_without_multistart():
    """A seeded random initial guess currently means one deterministic solve."""
    robot = Robot((200, 0, 0, q1), (200, 0, 0, q2))
    results = [
        solve_position_ik(robot, [200, 200, 0], q0=None, random_state=13)
        for _ in range(2)
    ]
    assert results[0].converged == results[1].converged
    assert results[0].iterations == results[1].iterations
    assert results[0].message == results[1].message
    np.testing.assert_allclose(results[0].q, results[1].q, rtol=0, atol=0)
    assert results[0].error == results[1].error


def test_near_singular_position_ik_baseline():
    """Retain near-singularity behavior as a control for the later correction."""
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    solution = solve_position_ik(
        robot, [1.99, 0, 0], q0=[1, 0],
        damping=10, tol=1e-8, max_iter=200,
    )
    assert solution.converged
    assert solution.error < 1e-8


def test_pose_ik_reachable_target_remains_solved():
    """Pose IK shares LM machinery but mixes orientation and position units."""
    from moro.inverse_kinematics import solve_pose_ik

    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    target = robot.T.subs({q1: 0.55, q2: 0.35})
    result = solve_pose_ik(
        robot, target, q0=[0.5, 0.3],
        position_tol=1e-8, orientation_tol=1e-8, max_iter=200,
    )
    assert result.converged
    assert result.position_error < 1e-8
    assert result.orientation_error < 1e-8


@pytest.mark.parametrize("max_iter", [1, 3, 5])
def test_lm_rejected_attempts_still_count_toward_max_iter(max_iter):
    """Damping adjustments cannot bypass the public iteration budget."""
    robot = Robot((200, 0, 0, q1), (200, 0, 0, q2))
    solution = solve_position_ik(
        robot, [200, 200, 0], q0=[0.1, 0.1],
        max_iter=max_iter,
    )
    assert not solution.converged
    assert solution.iterations == max_iter
    assert solution.message == "Maximum number of iterations reached."


def test_lm_stagnation_one_does_not_stop_at_first_damping_rejection():
    """One rejected, still-movable LM trial is not an accepted stagnation step."""
    robot = Robot((200, 0, 0, q1), (200, 0, 0, q2))
    solution = solve_position_ik(
        robot, [200, 200, 0], q0=[0.1, 0.1],
        stagnation_iterations=1,
    )
    assert solution.converged
    assert solution.error < 1e-6


@pytest.mark.parametrize("stagnation_iterations", [5, 6])
def test_default_damping_scale_pair_for_reported_rr_rrr(regression_case, stagnation_iterations):
    """A's damping-rejection fix should solve the reported m/mm counterparts.

    Use physically equivalent absolute position thresholds. These assertions
    test the reported examples only; they are *not* a general scale-invariance
    guarantee for arbitrary robots or initial configurations.
    """
    make_robot, target, q0 = regression_case
    for scale in (1.0, 1e-3):
        solution = solve_position_ik(
            make_robot(scale),
            target * scale,
            q0=q0,
            tol=1e-6 * scale,
            error_change_tol=1e-12 * scale,
            stagnation_iterations=stagnation_iterations,
            max_iter=100,
        )
        assert solution.converged, (
            f"scale={scale} stalled={stagnation_iterations} "
            f"error={solution.error} iterations={solution.iterations} "
            f"reason={solution.message}"
        )
        assert solution.error < 1e-6 * scale


def test_unreachable_scaled_rr_stays_nonconvergent():
    """The relaxed LM rejection accounting must not claim unreachable goals."""
    for scale in (1.0, 1e-3):
        robot = Robot((200 * scale, 0, 0, q1),
                      (200 * scale, 0, 0, q2))
        solution = solve_position_ik(
            robot,
            [1000 * scale, 1000 * scale, 0],
            q0=[0.1, 0.1],
            tol=1e-6 * scale,
            error_change_tol=1e-12 * scale,
            max_iter=100,
        )
        assert not solution.converged
        assert solution.iterations <= 100
        assert np.isfinite(solution.error)


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_scaled_rr_lm_with_restricted_revolute_joint(scale):
    """Limits can prevent reaching a Cartesian target at either scale."""
    robot = Robot((200 * scale, 0, 0, q1),
                  (200 * scale, 0, 0, q2))
    sol = solve_position_ik(
        robot, [200 * scale, 200 * scale, 0],
        q0=[0, 0],
        joint_limits=[(0, 0), (0, 0)],
        tol=1e-6 * scale,
        error_change_tol=1e-12 * scale,
        max_iter=50,
    )
    assert not sol.converged
    assert sol.q == [0.0, 0.0]
    assert sol.iterations <= 50
