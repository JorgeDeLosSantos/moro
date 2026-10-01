"""Hardening tests for Moro 0.5-F numerical dynamics."""

import numpy as np
import pytest
import sympy as sp
from matplotlib.animation import FuncAnimation

from moro.abc import q1, q2, t as symbolic_time
from moro.core import Robot
from moro.dynamics import forward_dynamics, inverse_dynamics, simulate
from moro.trajectory import joint_trajectory
from moro.visualization import RobotVisualizer


def make_mixed_rp_robot():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.0, 0, q2, 0, "p"),
    )
    robot.masses = [1.0, 0.8]
    robot.cm_positions = [
        (-0.5, 0, 0),
        (0, 0, 0),
    ]
    robot.inertia_tensors = [
        sp.diag(0, 0, 0.05),
        sp.diag(0.01, 0.01, 0.01),
    ]
    robot.gravity = (0, 0, 0)
    return robot


def make_planar_2r_zero_gravity():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.8, 0, 0, q2, "r"),
    )
    robot.masses = [1.2, 0.8]
    robot.cm_positions = [
        (-0.5, 0, 0),
        (-0.4, 0, 0),
    ]
    robot.inertia_tensors = [
        sp.diag(0, 0, 0.08),
        sp.diag(0, 0, 0.04),
    ]
    robot.gravity = (0, 0, 0)
    return robot


def make_pendulum():
    robot = Robot((1.0, 0, 0, q1, "r"))
    robot.masses = [1.0]
    robot.cm_positions = [(-0.5, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, 0.02)]
    robot.gravity = (0, -9.81, 0)
    return robot


def test_mixed_revolute_prismatic_inverse_forward_consistency():
    robot = make_mixed_rp_robot()
    q = np.array([0.35, 0.2])
    qd = np.array([0.4, -0.3])
    qdd_ref = np.array([-0.6, 0.8])

    tau = inverse_dynamics(robot, q, qd, qdd_ref)
    qdd = forward_dynamics(robot, q, qd, tau)

    assert tau.shape == (2,)
    assert qdd.shape == (2,)
    np.testing.assert_allclose(qdd, qdd_ref, rtol=1e-9, atol=1e-10)


def test_coriolis_only_term_matches_symbolic_model_and_is_nontrivial():
    robot = make_planar_2r_zero_gravity()
    q = np.array([0.4, -0.7])
    qd = np.array([0.8, -0.45])

    tau = inverse_dynamics(robot, q, qd, [0.0, 0.0])

    substitutions = dict(zip(robot.qs, q))
    substitutions.update(
        dict(zip([qi.diff(symbolic_time) for qi in robot.qs], qd))
    )
    C = np.asarray(
        robot.coriolis_matrix().subs(substitutions),
        dtype=float,
    )
    expected = C @ qd

    assert np.linalg.norm(expected) > 1e-6
    np.testing.assert_allclose(tau, expected, rtol=1e-10, atol=1e-10)


def test_gravity_driven_pendulum_moves_from_rest():
    robot = make_pendulum()
    t_eval = np.linspace(0.0, 0.4, 21)

    initial_qdd = forward_dynamics(robot, 0.0, 0.0, 0.0)[0]
    sol = simulate(
        robot,
        (0.0, 0.4),
        0.0,
        qd0=0.0,
        tau=None,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    assert initial_qdd < 0.0
    assert sol.qdd[0, 0] == pytest.approx(initial_qdd)
    assert sol.q[-1, 0] < sol.q[0, 0]
    assert abs(sol.qd[-1, 0]) > 1e-3


def test_conservative_pendulum_approximately_conserves_mechanical_energy():
    robot = make_pendulum()
    t_eval = np.linspace(0.0, 2.0, 101)

    sol = simulate(
        robot,
        (0.0, 2.0),
        0.35,
        qd0=0.0,
        tau=None,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
    )

    K = robot.kinetic_energy()[0]
    P = robot.potential_energy()[0]
    qd1 = q1.diff(symbolic_time)

    energies = np.array([
        float((K + P).subs({q1: q, qd1: qd}))
        for q, qd in zip(sol.q[:, 0], sol.qd[:, 0])
    ])

    scale = max(1.0, abs(energies[0]))
    relative_drift = np.max(np.abs(energies - energies[0])) / scale
    assert relative_drift < 1e-7


def test_symbolic_geometry_mass_inertia_and_gravity_parameters_work_end_to_end():
    l, lc, m, iz, g = sp.symbols("l lc m iz g", positive=True)
    robot = Robot((l, 0, 0, q1, "r"))
    robot.masses = [m]
    robot.cm_positions = [(-lc, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, iz)]
    robot.gravity = (0, -g, 0)

    params = {
        l: 1.0,
        lc: 0.5,
        m: 2.0,
        iz: 0.1,
        g: 9.81,
    }
    q = 0.25
    qd = -0.2
    qdd_ref = 0.4

    tau = inverse_dynamics(
        robot,
        q,
        qd,
        qdd_ref,
        parameters=params,
    )
    qdd = forward_dynamics(
        robot,
        q,
        qd,
        tau,
        parameters=params,
    )
    sol = simulate(
        robot,
        (0.0, 0.1),
        q,
        qd0=qd,
        tau=tau,
        parameters=params,
        t_eval=[0.0, 0.05, 0.1],
    )

    np.testing.assert_allclose(qdd, [qdd_ref], rtol=1e-10, atol=1e-10)
    assert sol.success is True
    assert robot.masses == [m]
    assert robot.cm_positions[0] == sp.Matrix([-lc, 0, 0])
    assert robot.gravity == sp.Matrix([0, -g, 0])


def test_joint_trajectory_can_feed_samplewise_inverse_dynamics():
    robot = make_planar_2r_zero_gravity()
    traj = joint_trajectory(
        [0.1, -0.2],
        [0.5, 0.35],
        np.linspace(0.0, 1.0, 5),
        method="quintic",
    )

    tau = np.array([
        inverse_dynamics(robot, q, qd, qdd)
        for q, qd, qdd in zip(traj.q, traj.qd, traj.qdd)
    ])

    assert tau.shape == (traj.samples, traj.dof)
    assert np.all(np.isfinite(tau))


def test_simulation_joint_matrix_is_directly_visualizable():
    robot = make_pendulum()
    sol = simulate(
        robot,
        (0.0, 0.1),
        0.1,
        qd0=0.0,
        tau=None,
        t_eval=[0.0, 0.05, 0.1],
    )

    viz = RobotVisualizer(robot)
    animation = viz.animate(sol.q, backend="matplotlib")

    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = True
