"""Tests for Moro numerical dynamics and simulation."""

import types

import numpy as np
import pytest
import sympy as sp

import moro.dynamics as dynamics_module
from moro.abc import q1, q2
from moro.core import Robot
from moro.dynamics import (
    DynamicsSolution,
    forward_dynamics,
    inverse_dynamics,
    simulate,
    state_derivative,
)


def make_prismatic_1dof(*, mass=2.0, gravity=(0, 0, 0)):
    robot = Robot((0, 0, q1, 0, "p"))
    robot.masses = [mass]
    robot.cm_positions = [(0, 0, 0)]
    robot.inertia_tensors = [sp.zeros(3)]
    robot.gravity = gravity
    return robot


def make_planar_2r():
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
    robot.gravity = (0, -9.81, 0)
    return robot


class TestDynamicsSolution:
    def test_valid_result_and_properties(self):
        sol = DynamicsSolution(
            t=[1.0, 1.5, 2.0],
            q=[[0.0], [0.1], [0.2]],
            qd=[[0.0], [0.2], [0.4]],
            qdd=[[0.4], [0.4], [0.4]],
            success=True,
            message="ok",
            method="RK45",
        )

        assert sol.samples == 3
        assert sol.duration == pytest.approx(1.0)
        assert sol.dof == 1
        assert "DynamicsSolution" in repr(sol)
        assert "samples=3" in repr(sol)
        assert "RK45" in repr(sol)

    def test_defensive_copying(self):
        t = np.array([0.0, 1.0])
        q = np.array([[0.0], [1.0]])
        qd = np.array([[0.0], [1.0]])
        qdd = np.array([[1.0], [1.0]])

        sol = DynamicsSolution(t, q, qd, qdd, True, "ok", "RK45")
        t[:] = -1
        q[:] = -1

        np.testing.assert_allclose(sol.t, [0.0, 1.0])
        np.testing.assert_allclose(sol.q[:, 0], [0.0, 1.0])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="identical shapes"):
            DynamicsSolution(
                [0.0, 1.0],
                [[0.0], [1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0], [1.0]],
                True,
                "ok",
                "RK45",
            )


class TestNumericalDynamicsValidation:
    def test_static_joint_variables_are_rejected(self):
        qs = sp.symbols("q")
        robot = Robot((0, 0, qs, 0, "p"))
        robot.masses = [1.0]
        robot.cm_positions = [(0, 0, 0)]
        robot.inertia_tensors = [sp.zeros(3)]
        robot.gravity = (0, 0, 0)

        with pytest.raises(ValueError, match="time-dependent"):
            forward_dynamics(robot, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, 1 + 2j])
    def test_rejects_invalid_state_values(self, bad):
        robot = make_prismatic_1dof()
        with pytest.raises(ValueError):
            inverse_dynamics(robot, bad, 0.0, 0.0)

    def test_scalar_input_rejected_for_multidof(self):
        robot = make_planar_2r()
        with pytest.raises(ValueError, match="broadcasting"):
            forward_dynamics(robot, 0.1, [0.0, 0.0], [0.0, 0.0])

    def test_unresolved_parameter_is_rejected(self):
        m = sp.symbols("m", positive=True)
        robot = make_prismatic_1dof(mass=m)

        with pytest.raises(ValueError, match="no numerical value"):
            forward_dynamics(robot, 0.0, 0.0, 0.0)

    def test_parameter_substitution_does_not_mutate_robot(self):
        m = sp.symbols("m", positive=True)
        robot = make_prismatic_1dof(mass=m)

        tau = inverse_dynamics(
            robot,
            0.0,
            0.0,
            1.5,
            parameters={m: 2.0},
        )

        np.testing.assert_allclose(tau, [3.0])
        assert robot.masses == [m]


class TestInverseForwardDynamics:
    def test_1dof_scalar_inputs_and_output_shape(self):
        robot = make_prismatic_1dof(mass=2.0)

        tau = inverse_dynamics(robot, 0.3, -0.2, 1.5)
        qdd = forward_dynamics(robot, 0.3, -0.2, tau)

        assert tau.shape == (1,)
        assert qdd.shape == (1,)
        np.testing.assert_allclose(tau, [3.0], atol=1e-12)
        np.testing.assert_allclose(qdd, [1.5], atol=1e-12)

    def test_planar_2r_inverse_forward_consistency(self):
        robot = make_planar_2r()
        q = np.array([0.3, -0.4])
        qd = np.array([0.5, -0.2])
        qdd_ref = np.array([0.7, -0.6])

        tau = inverse_dynamics(robot, q, qd, qdd_ref)
        qdd = forward_dynamics(robot, q, qd, tau)

        np.testing.assert_allclose(qdd, qdd_ref, rtol=1e-9, atol=1e-10)

    def test_gravity_only_matches_gravity_vector(self):
        robot = make_planar_2r()
        q = np.array([0.4, -0.3])

        tau = inverse_dynamics(
            robot,
            q,
            [0.0, 0.0],
            [0.0, 0.0],
        )

        G = np.asarray(
            robot.gravity_vector().subs(dict(zip(robot.qs, q))),
            dtype=float,
        ).reshape(-1)

        np.testing.assert_allclose(tau, G, rtol=1e-10, atol=1e-10)

    def test_inertia_only_case(self):
        robot = make_prismatic_1dof(mass=3.0)
        tau = inverse_dynamics(robot, 0.2, 0.0, 2.0)
        np.testing.assert_allclose(tau, [6.0], atol=1e-12)

    def test_singular_mass_matrix_raises(self):
        robot = make_prismatic_1dof(mass=0.0)

        with pytest.raises(np.linalg.LinAlgError, match="Mass matrix is singular"):
            forward_dynamics(robot, 0.0, 0.0, 1.0)


class TestStateDerivative:
    def test_state_ordering_and_forward_dynamics_consistency(self):
        robot = make_prismatic_1dof(mass=2.0)
        state = np.array([0.3, -0.4])

        derivative = state_derivative(
            robot,
            1.2,
            state,
            tau=2.0,
        )

        np.testing.assert_allclose(derivative, [-0.4, 1.0])

    def test_none_tau_equals_zero_tau(self):
        robot = make_prismatic_1dof(mass=2.0)
        state = [0.2, 0.3]

        a = state_derivative(robot, 0.0, state, tau=None)
        b = state_derivative(robot, 0.0, state, tau=0.0)

        np.testing.assert_allclose(a, b)

    def test_time_dependent_callable(self):
        robot = make_prismatic_1dof(mass=2.0)

        def force(t, q, qd):
            return 4.0 * t

        derivative = state_derivative(
            robot,
            0.5,
            [0.0, 0.0],
            tau=force,
        )

        np.testing.assert_allclose(derivative, [0.0, 1.0])

    def test_state_feedback_callable_receives_arrays(self):
        robot = make_prismatic_1dof(mass=1.0)
        seen = {}

        def control(t, q, qd):
            seen["q_shape"] = q.shape
            seen["qd_shape"] = qd.shape
            return -2.0 * q - 0.5 * qd

        state_derivative(robot, 0.0, [0.5, -0.2], tau=control)

        assert seen["q_shape"] == (1,)
        assert seen["qd_shape"] == (1,)

    def test_user_callable_exception_is_preserved(self):
        robot = make_prismatic_1dof()

        def bad_force(t, q, qd):
            raise RuntimeError("controller failed")

        with pytest.raises(RuntimeError, match="controller failed"):
            state_derivative(robot, 0.0, [0.0, 0.0], tau=bad_force)

    def test_invalid_callable_return_is_rejected(self):
        robot = make_planar_2r()

        def bad_force(t, q, qd):
            return 1.0

        with pytest.raises(ValueError, match="broadcasting"):
            state_derivative(
                robot,
                0.0,
                [0.0, 0.0, 0.0, 0.0],
                tau=bad_force,
            )


class TestSimulationValidation:
    @pytest.mark.parametrize(
        "t_span",
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, np.inf),
            (0.0,),
            (0.0, 1.0, 2.0),
        ],
    )
    def test_rejects_invalid_t_span(self, t_span):
        robot = make_prismatic_1dof()
        with pytest.raises(ValueError):
            simulate(robot, t_span, 0.0)

    @pytest.mark.parametrize(
        "t_eval",
        [
            [0.0, 0.5, 0.5, 1.0],
            [0.0, 0.8, 0.4],
            [-0.1, 0.5, 1.0],
            [0.0, 0.5, 1.1],
            [0.0, np.nan, 1.0],
            [[0.0, 1.0]],
        ],
    )
    def test_rejects_invalid_t_eval(self, t_eval):
        robot = make_prismatic_1dof()
        with pytest.raises(ValueError):
            simulate(robot, (0.0, 1.0), 0.0, t_eval=t_eval)

    def test_qd0_none_matches_zero(self):
        robot = make_prismatic_1dof()
        t_eval = np.linspace(0.0, 1.0, 11)

        a = simulate(robot, (0.0, 1.0), 0.0, qd0=None, t_eval=t_eval)
        b = simulate(robot, (0.0, 1.0), 0.0, qd0=0.0, t_eval=t_eval)

        np.testing.assert_allclose(a.q, b.q)
        np.testing.assert_allclose(a.qd, b.qd)


class TestSimulation:
    def test_analytical_constant_acceleration(self):
        robot = make_prismatic_1dof(mass=2.0)
        t0 = 2.0
        tf = 4.0
        t_eval = np.linspace(t0, tf, 21)
        q0 = 0.3
        qd0 = -0.2
        tau = 4.0
        acceleration = 2.0

        sol = simulate(
            robot,
            (t0, tf),
            q0,
            qd0=qd0,
            tau=tau,
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )

        elapsed = t_eval - t0
        q_expected = q0 + qd0 * elapsed + 0.5 * acceleration * elapsed**2
        qd_expected = qd0 + acceleration * elapsed

        assert sol.success is True
        np.testing.assert_allclose(sol.t, t_eval)
        np.testing.assert_allclose(sol.q[:, 0], q_expected, atol=1e-8)
        np.testing.assert_allclose(sol.qd[:, 0], qd_expected, atol=1e-8)
        np.testing.assert_allclose(sol.qdd[:, 0], acceleration, atol=1e-10)

    def test_nonuniform_t_eval(self):
        robot = make_prismatic_1dof()
        t_eval = np.array([0.0, 0.05, 0.2, 0.7, 1.0])

        sol = simulate(
            robot,
            (0.0, 1.0),
            0.0,
            tau=1.0,
            t_eval=t_eval,
        )

        np.testing.assert_allclose(sol.t, t_eval)

    def test_qdd_reconstruction_matches_forward_dynamics(self):
        robot = make_prismatic_1dof(mass=2.0)
        sol = simulate(
            robot,
            (0.0, 1.0),
            0.0,
            tau=3.0,
            t_eval=np.linspace(0.0, 1.0, 9),
        )

        for q, qd, qdd in zip(sol.q, sol.qd, sol.qdd):
            ref = forward_dynamics(robot, q, qd, [3.0])
            np.testing.assert_allclose(qdd, ref)

    def test_joint_limits_are_not_enforced(self):
        robot = make_prismatic_1dof(mass=1.0)
        robot.joint_limits = [(0.0, 0.1)]

        sol = simulate(
            robot,
            (0.0, 1.0),
            0.0,
            tau=2.0,
            t_eval=np.linspace(0.0, 1.0, 11),
        )

        assert sol.q[-1, 0] > 0.1

    def test_pd_like_feedback_moves_toward_reference(self):
        robot = make_prismatic_1dof(mass=1.0)
        q_ref = 1.0

        def control(t, q, qd):
            return -8.0 * (q - q_ref) - 4.0 * qd

        sol = simulate(
            robot,
            (0.0, 3.0),
            0.0,
            tau=control,
            t_eval=np.linspace(0.0, 3.0, 61),
        )

        assert abs(sol.q[-1, 0] - q_ref) < abs(sol.q[0, 0] - q_ref)


def test_simulate_returns_failed_partial_solution(monkeypatch):
    robot = make_prismatic_1dof(mass=1.0)

    fake = types.SimpleNamespace(
        t=np.array([0.0, 0.2]),
        y=np.array([
            [0.0, 0.02],
            [0.0, 0.2],
        ]),
        success=False,
        message="controlled failure",
    )

    monkeypatch.setattr(dynamics_module, "solve_ivp", lambda *args, **kwargs: fake)

    sol = simulate(
        robot,
        (0.0, 1.0),
        0.0,
        tau=1.0,
    )

    assert sol.success is False
    assert sol.message == "controlled failure"
    assert sol.samples == 2
    np.testing.assert_allclose(sol.qdd[:, 0], 1.0)


def test_solver_options_are_forwarded_only_when_explicit(monkeypatch):
    robot = make_prismatic_1dof(mass=1.0)
    calls = []

    def fake_solve_ivp(fun, t_span, y0, method, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            t=np.array([t_span[0], t_span[1]]),
            y=np.column_stack((y0, y0)),
            success=True,
            message="ok",
        )

    monkeypatch.setattr(dynamics_module, "solve_ivp", fake_solve_ivp)

    simulate(robot, (0.0, 1.0), 0.0)
    assert "rtol" not in calls[-1]
    assert "atol" not in calls[-1]
    assert "max_step" not in calls[-1]

    simulate(
        robot,
        (0.0, 1.0),
        0.0,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.1,
    )
    assert calls[-1]["rtol"] == pytest.approx(1e-6)
    assert calls[-1]["atol"] == pytest.approx(1e-8)
    assert calls[-1]["max_step"] == pytest.approx(0.1)
