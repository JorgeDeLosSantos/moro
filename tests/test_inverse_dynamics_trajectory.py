"""API-002/B1 contract and core inverse-dynamics history integration tests."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

import moro.dynamics as dynamics_module
from moro.abc import q1, q2
from moro.core import Robot
from moro.dynamics import (
    DynamicsSolution,
    inverse_dynamics,
    inverse_dynamics_trajectory,
)
from moro.trajectory import joint_trajectory


def prismatic_robot(*, mass=2.0):
    robot = Robot((0, 0, q1, 0, "p"))
    robot.masses = [mass]
    robot.cm_positions = [(0, 0, 0)]
    robot.inertia_tensors = [sp.zeros(3)]
    robot.gravity = (0, 0, 0)
    return robot


def planar_rr_robot():
    robot = Robot((1.0, 0, 0, q1, "r"), (0.8, 0, 0, q2, "r"))
    robot.masses = [1.2, 0.8]
    robot.cm_positions = [(-0.5, 0, 0), (-0.4, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, 0.08), sp.diag(0, 0, 0.04)]
    robot.gravity = (0, -9.81, 0)
    return robot


def test_1dof_analytical_force_and_time_optional():
    robot = prismatic_robot(mass=2.0)
    history = SimpleNamespace(
        q=np.array([[0.0], [0.1], [0.2]]),
        qd=np.array([[0.2], [0.3], [0.4]]),
        qdd=np.array([[1.0], [-0.5], [0.0]]),
    )
    result = inverse_dynamics_trajectory(robot, history)
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result[:, 0], [2.0, -1.0, 0.0])


def test_jointtrajectory_matches_single_state_inverse_dynamics():
    robot = prismatic_robot()
    traj = joint_trajectory(
        0.0, 1.0, np.linspace(0.0, 2.0, 6), method="quintic"
    )
    actual = inverse_dynamics_trajectory(robot, traj)
    expected = np.vstack([
        inverse_dynamics(robot, q, qd, qdd)
        for q, qd, qdd in zip(traj.q, traj.qd, traj.qdd)
    ])
    assert actual.shape == (traj.samples, robot.dof)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_dynamicssolution_and_one_sample_history():
    robot = prismatic_robot()
    solution = DynamicsSolution(
        t=[0.5],
        q=[[0.3]],
        qd=[[0.2]],
        qdd=[[-0.7]],
        success=True,
        message="ok",
        method="RK45",
    )
    np.testing.assert_allclose(
        inverse_dynamics_trajectory(robot, solution), [[-1.4]]
    )


def test_symbolic_dynamics_parameter_and_model_compiled_once():
    m = sp.symbols("m", positive=True)
    robot = prismatic_robot(mass=m)
    history = SimpleNamespace(
        t=[-1.0, -0.5, 0.5],
        q=[[0.0], [0.2], [0.4]],
        qd=[[0.0], [0.3], [0.6]],
        qdd=[[0.5], [-0.5], [1.0]],
    )
    with patch.object(
        dynamics_module,
        "_prepare_numerical_model",
        wraps=dynamics_module._prepare_numerical_model,
    ) as prepare:
        result = inverse_dynamics_trajectory(robot, history, parameters={m: 3.0})
    prepare.assert_called_once_with(robot, parameters={m: 3.0})
    np.testing.assert_allclose(result[:, 0], [1.5, -1.5, 3.0])


def test_rr_matches_ideal_per_state_including_gravity_and_coriolis():
    robot = planar_rr_robot()
    motion = SimpleNamespace(
        q=[[0.0, 0.0], [0.2, -0.1], [0.4, 0.3]],
        qd=[[0.1, 0.0], [0.3, -0.2], [0.2, 0.4]],
        qdd=[[0.0, 0.1], [0.2, -0.3], [-0.5, 0.2]],
    )
    actual = inverse_dynamics_trajectory(robot, motion)
    expected = np.vstack([
        inverse_dynamics(robot, q, qd, qdd)
        for q, qd, qdd in zip(motion.q, motion.qd, motion.qdd)
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_input_histories_are_not_mutated_and_output_is_independent():
    robot = prismatic_robot()
    q = np.array([[0.0], [0.3]])
    qd = np.array([[0.1], [0.5]])
    qdd = np.array([[0.2], [-0.3]])
    t = np.array([0.0, 0.1])
    originals = [array.copy() for array in (q, qd, qdd, t)]
    motion = SimpleNamespace(q=q, qd=qd, qdd=qdd, t=t)

    tau = inverse_dynamics_trajectory(robot, motion)
    for actual, expected in zip((q, qd, qdd, t), originals):
        np.testing.assert_array_equal(actual, expected)
    tau[0, 0] = 100.0
    assert motion.qdd[0, 0] == 0.2


@pytest.mark.parametrize("motion,pattern", [
    (object(), "missing 'q'"),
    (SimpleNamespace(q=[[0.0]]), "missing 'qd'"),
    (SimpleNamespace(q=[[0.0]], qd=[[0.0]]), "missing 'qdd'"),
    (SimpleNamespace(q=[], qd=[], qdd=[]), "2-dimensional"),
    (
        SimpleNamespace(q=np.empty((0, 1)), qd=np.empty((0, 1)),
                        qdd=np.empty((0, 1))),
        "at least one sample",
    ),
    (
        SimpleNamespace(q=[[0.0]], qd=[[0.0], [0.0]], qdd=[[0.0]]),
        "identical shapes",
    ),
    (
        SimpleNamespace(q=[[0.0, 0.0]], qd=[[0.0, 0.0]],
                        qdd=[[0.0, 0.0]]),
        "robot.dof=1",
    ),
    (
        SimpleNamespace(q=[[0.0]], qd=[[np.inf]], qdd=[[0.0]]),
        "motion.qd",
    ),
    (
        SimpleNamespace(q=[[0.0]], qd=[[0.0]], qdd=[[1.0 + 1.0j]]),
        "motion.qdd",
    ),
    (
        SimpleNamespace(q=[[0.0]], qd=[[0.0]], qdd=[[0.0]], t=[0.0, 0.1]),
        "one timestamp per motion sample",
    ),
    (
        SimpleNamespace(q=[[0.0], [0.0]], qd=[[0.0], [0.0]],
                        qdd=[[0.0], [0.0]], t=[1.0, 1.0]),
        "strictly increasing",
    ),
    (
        SimpleNamespace(q=[[0.0]], qd=[[0.0]], qdd=[[0.0]], t=[np.nan]),
        "motion.t",
    ),
])
def test_reject_bad_histories_before_model_preparation(motion, pattern):
    robot = prismatic_robot()
    with patch.object(dynamics_module, "_prepare_numerical_model") as prepare:
        with pytest.raises(ValueError, match=pattern):
            inverse_dynamics_trajectory(robot, motion)
        prepare.assert_not_called()


def test_sample_evaluation_error_has_zero_based_index():
    robot = prismatic_robot()
    motion = SimpleNamespace(
        q=[[0.0], [0.1], [0.2]],
        qd=[[0.0], [0.0], [0.0]],
        qdd=[[0.0], [0.0], [0.0]],
    )

    counter = 0

    def broken_row(model, q, qd, qdd):
        nonlocal counter
        index = counter
        counter += 1
        if index == 1:
            raise ValueError("custom numerical singularity")
        return np.zeros(model.dof)

    with patch.object(
        dynamics_module, "_inverse_dynamics_prepared", side_effect=broken_row
    ):
        with pytest.raises(ValueError, match="sample 1: custom numerical singularity"):
            inverse_dynamics_trajectory(robot, motion)
