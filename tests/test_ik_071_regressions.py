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


@pytest.mark.xfail(
    strict=True,
    reason="0.7.0 LM counts five rejected damping adjustments as stagnation",
)
def test_lm_default_does_not_terminate_before_first_accepted_step(regression_case):
    make_robot, target, q0 = regression_case
    solution = solve_position_ik(make_robot(1.0), target, q0=q0)

    # Currently XFAIL: the reported cases exit after five rejected LM trials.
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
