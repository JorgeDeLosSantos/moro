"""API-002/B2: physical model coverage and regression contracts.

These tests deliberately check numerical equivalence and preparation counts
instead of performance measured by unreliable wall-clock timings.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

import moro.dynamics as dynamics
from moro.abc import q1, q2
from moro.core import Robot
from moro.dynamics import (
    DynamicsSolution,
    forward_dynamics,
    inverse_dynamics,
    inverse_dynamics_trajectory,
)
from moro.trajectory import joint_trapezoidal_trajectory, joint_trajectory


def revolute_robot():
    """A single rotor with a mass at a 1 m radius, zero gravity."""
    robot = Robot((0, 0, 0, q1, "r"))
    robot.masses = [2.0]
    robot.cm_positions = [(1.0, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, sp.Rational(1, 4))]
    robot.gravity = (0, 0, 0)
    return robot


def rr_robot():
    robot = Robot((0.9, 0, 0, q1, "r"), (0.6, 0, 0, q2, "r"))
    robot.masses = [1.5, 1.0]
    robot.cm_positions = [(-0.45, 0, 0), (-0.3, 0, 0)]
    robot.inertia_tensors = [
        sp.diag(0, 0, 0.06),
        sp.diag(0, 0, 0.025),
    ]
    robot.gravity = (0, -9.81, 0)
    return robot


def rp_robot():
    """Rotation about Z followed by prismatic motion along Z."""
    robot = Robot((0.8, 0, 0, q1, "r"), (0, 0, q2, 0, "p"))
    robot.masses = [1.2, 0.7]
    robot.cm_positions = [(-0.3, 0, 0), (0, 0, 0)]
    robot.inertia_tensors = [
        sp.diag(0.01, 0.01, 0.12),
        sp.diag(0.02, 0.02, 0.06),
    ]
    robot.gravity = (0, 0, -9.81)
    return robot


@pytest.mark.parametrize("model", [revolute_robot, rr_robot, rp_robot], ids=["R", "RR", "RP"])
def test_samplewise_equivalence_and_forward_dynamics(model):
    robot = model()
    dof = robot.dof
    q = np.array([
        np.linspace(0.1, 0.3, dof),
        np.linspace(-0.2, 0.4, dof),
        np.linspace(0.6, -0.1, dof),
    ])
    qd = np.array([
        np.linspace(0.2, -0.1, dof),
        np.linspace(-0.3, 0.4, dof),
        np.linspace(0.5, 0.2, dof),
    ])
    qdd = np.array([
        np.linspace(0.7, -0.3, dof),
        np.linspace(-0.5, 0.8, dof),
        np.linspace(0.0, 0.1, dof),
    ])
    motion = SimpleNamespace(q=q, qd=qd, qdd=qdd, t=[0.0, 0.25, 2.0])
    history = inverse_dynamics_trajectory(robot, motion)
    assert history.shape == (3, dof)
    assert history.dtype == np.dtype(float)
    assert np.all(np.isfinite(history))

    expected = np.vstack([
        inverse_dynamics(robot, q_i, qd_i, qdd_i)
        for q_i, qd_i, qdd_i in zip(q, qd, qdd)
    ])
    np.testing.assert_allclose(history, expected, rtol=1e-10, atol=1e-10)
    for q_i, qd_i, tau_i, qdd_i in zip(q, qd, history, qdd):
        np.testing.assert_allclose(
            forward_dynamics(robot, q_i, qd_i, tau_i),
            qdd_i, rtol=1e-10, atol=1e-10,
        )


def test_single_revolute_joint_has_analytic_inertia_solution():
    robot = revolute_robot()
    qdd = np.array([[0.0], [1.5], [-0.4], [2.0]])
    motion = SimpleNamespace(
        q=np.array([[0.0], [0.25], [1.0], [-0.5]]),
        qd=np.array([[0.0], [0.4], [-0.7], [1.5]]),
        qdd=qdd,
    )
    # M = m * r**2 + Izz = 2 * 1**2 + 0.25 = 2.25 kg m^2.
    tau = inverse_dynamics_trajectory(robot, motion)
    np.testing.assert_allclose(tau, 2.25 * qdd, rtol=1e-12, atol=1e-12)


def test_symbolic_length_mass_inertia_and_gravity_do_not_mutate_model():
    L, m, Iz, g = sp.symbols("L m Iz g", positive=True)
    robot = Robot((L, 0, 0, q1, "r"))
    robot.masses = [m]
    robot.cm_positions = [(-L / 2, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, Iz)]
    robot.gravity = (0, -g, 0)
    motion = SimpleNamespace(
        q=[[0.0], [0.3], [-0.2]],
        qd=[[0.0], [0.5], [0.1]],
        qdd=[[0.4], [-0.1], [0.7]],
        t=[-3.0, -2.0, 0.0],
    )
    parameters = {L: 1.2, m: 2.0, Iz: 0.08, g: 9.81}
    initial_parameters = parameters.copy()
    result = inverse_dynamics_trajectory(robot, motion, parameters=parameters)
    expected = np.vstack([
        inverse_dynamics(robot, qi, qdi, qddi, parameters=parameters)
        for qi, qdi, qddi in zip(motion.q, motion.qd, motion.qdd)
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    assert parameters == initial_parameters
    assert robot.masses == [m]
    assert robot.inertia_tensors[0][2, 2] == Iz
    assert robot.cm_positions[0][0] == -L / 2


def test_missing_symbolic_parameter_fails_during_model_preparation():
    m = sp.Symbol("m", positive=True)
    robot = Robot((0, 0, q1, 0, "p"))
    robot.masses = [m]
    robot.cm_positions = [(0, 0, 0)]
    robot.inertia_tensors = [sp.zeros(3)]
    robot.gravity = (0, 0, 0)
    motion = SimpleNamespace(q=[[0], [1]], qd=[[0], [0]], qdd=[[1], [2]])
    with patch.object(dynamics, "_inverse_dynamics_prepared") as rows:
        with pytest.raises(ValueError, match="no numerical value: m"):
            inverse_dynamics_trajectory(robot, motion)
        rows.assert_not_called()


def test_trapezoidal_jointtrajectory_preserves_physical_sample_order():
    robot = revolute_robot()
    trajectory = joint_trapezoidal_trajectory(
        0.0, 1.1, v_max=0.7, a_max=1.0, samples=17,
    )
    result = inverse_dynamics_trajectory(robot, trajectory)
    assert result.shape == trajectory.q.shape
    np.testing.assert_allclose(result[:, 0], 2.25 * trajectory.qdd[:, 0])


def test_success_false_partial_dynamicssolution_is_supported():
    robot = revolute_robot()
    failed_partial = DynamicsSolution(
        t=[0.0, 0.2],
        q=[[0.0], [0.1]],
        qd=[[0.0], [0.3]],
        qdd=[[0.5], [-0.25]],
        success=False,
        message="integration failed after valid samples",
        method="RK45",
    )
    effort = inverse_dynamics_trajectory(robot, failed_partial)
    np.testing.assert_allclose(effort[:, 0], [1.125, -0.5625])


def test_timestamps_do_not_change_torque_when_state_derivatives_are_fixed():
    robot = revolute_robot()
    fields = {
        "q": [[0.0], [0.1], [0.2]],
        "qd": [[0.4], [0.5], [0.6]],
        "qdd": [[-0.2], [0.5], [0.0]],
    }
    early = SimpleNamespace(**fields, t=[0.0, 0.01, 2.5])
    late = SimpleNamespace(**fields, t=[30.0, 31.0, 32.0])
    no_time = SimpleNamespace(**fields)
    none_time = SimpleNamespace(**fields, t=None)
    expected = inverse_dynamics_trajectory(robot, no_time)
    for motion in (early, late, none_time):
        np.testing.assert_array_equal(
            inverse_dynamics_trajectory(robot, motion), expected,
        )


def test_preparation_and_lambdify_counts_do_not_scale_with_samples():
    robot = revolute_robot()
    n = 23
    motion = SimpleNamespace(
        q=np.zeros((n, 1)),
        qd=np.zeros((n, 1)),
        qdd=np.arange(n, dtype=float).reshape(n, 1),
    )
    with (
        patch.object(
            dynamics, "_prepare_numerical_model",
            wraps=dynamics._prepare_numerical_model,
        ) as prepare,
        patch.object(
            dynamics, "_inverse_dynamics_prepared",
            wraps=dynamics._inverse_dynamics_prepared,
        ) as evaluate,
        patch.object(
            dynamics.sp, "lambdify", wraps=dynamics.sp.lambdify,
        ) as lambdify,
    ):
        result = inverse_dynamics_trajectory(robot, motion)
    assert result.shape == (n, 1)
    prepare.assert_called_once_with(robot, parameters=None)
    assert evaluate.call_count == n
    assert lambdify.call_count == 3  # M, C and G compiled once each.


@pytest.mark.parametrize("name,bad", [
    ("q", [[0.0], [np.nan]]),
    ("qd", [[0.0], [np.inf]]),
    ("qdd", [[0.0], [-np.inf]]),
    ("q", [[0.0], [1j]]),
    ("qd", [[0.0], [0.5 + 0.3j]]),
    ("qdd", [[0.0], ["not-a-number"]]),
    ("q", [0.0, 0.2]),
    ("qd", [[[0.0]], [[0.1]]]),
])
def test_bad_state_values_rejected_without_compiling(name, bad):
    robot = revolute_robot()
    fields = {"q": [[0.0], [0.1]], "qd": [[0.0], [0.1]], "qdd": [[0.0], [0.1]]}
    fields[name] = bad
    motion = SimpleNamespace(**fields)
    with patch.object(dynamics, "_prepare_numerical_model") as prepare:
        with pytest.raises(ValueError, match="motion." + name):
            inverse_dynamics_trajectory(robot, motion)
        prepare.assert_not_called()


@pytest.mark.parametrize("bad_time", [
    [[0.0], [1.0]],
    [0.0, np.inf],
    [2.0, 1.0],
    [1.0, 1.0],
    [1j, 2.0],
])
def test_bad_timestamps_rejected_without_compiling(bad_time):
    robot = revolute_robot()
    motion = SimpleNamespace(
        q=[[0.0], [0.1]], qd=[[0.0], [0.1]], qdd=[[0.0], [0.1]],
        t=bad_time,
    )
    with patch.object(dynamics, "_prepare_numerical_model") as prepare:
        with pytest.raises(ValueError, match="motion.t"):
            inverse_dynamics_trajectory(robot, motion)
        prepare.assert_not_called()


def test_finite_input_overflow_reports_failing_sample_index():
    robot = revolute_robot()
    motion = SimpleNamespace(
        q=[[0.0], [0.1], [0.2]],
        qd=[[0.0], [0.0], [0.0]],
        qdd=[[0.0], [0.0], [1e308]],
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="Inverse dynamics failed at sample 2"):
            inverse_dynamics_trajectory(robot, motion)
