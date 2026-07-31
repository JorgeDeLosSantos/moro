"""
Tests for the inverse kinematics module.
"""
import pytest
import sympy as sp
import numpy as np
from moro.core import Robot
from moro.inverse_kinematics import solve_position_ik, IKSolution
from moro.abc import q1, q2


def end_effector_position(robot, q_values, parameters=None):
    """Evaluate the end-effector position for a numerical joint configuration."""
    substitutions = dict(zip(robot.qs, q_values))
    if parameters is not None:
        substitutions.update(parameters)

    position = robot.T[:3, 3].subs(substitutions)
    return np.asarray(position, dtype=float).reshape(3)


def assert_reaches_target(robot, solution, target, *, parameters=None, atol=1e-6):
    """Assert that the FK position associated with an IK solution reaches target."""
    position = end_effector_position(robot, solution.q, parameters=parameters)
    np.testing.assert_allclose(position, target, atol=atol)


def assert_within_joint_limits(q_values, limits):
    """Assert that every joint value lies within its corresponding bounds."""
    for q_value, (lower, upper) in zip(q_values, limits):
        assert lower <= q_value <= upper


class TestIKSolution:
    """Tests for the IKSolution data class."""

    def test_creation_and_attributes(self):
        sol = IKSolution([0.5, 1.2], converged=True, iterations=5, error=1e-8)
        assert sol.q == [0.5, 1.2]
        assert sol.converged is True
        assert sol.iterations == 5
        assert sol.error == 1e-8
        assert sol.method == "lm"

    def test_repr_converged(self):
        sol = IKSolution([0.5], converged=True, iterations=3, error=1e-6)
        assert "Converged" in repr(sol)
        assert "method" in repr(sol)

    def test_repr_not_converged(self):
        sol = IKSolution([0.5], converged=False, iterations=100, error=0.5)
        assert "Did not converge" in repr(sol)

    def test_scalar_q_converted_to_list(self):
        sol = IKSolution(0.5, converged=True, iterations=1, error=0.0)
        assert sol.q == [0.5]

    def test_custom_method_in_repr(self):
        sol = IKSolution([0.2], converged=True, iterations=4, error=1e-8, method="newton")
        assert "newton" in repr(sol)

    def test_ccd_method_in_repr(self):
        sol = IKSolution([0.2], converged=True, iterations=4, error=1e-8, method="ccd")
        assert "ccd" in repr(sol)


class TestSolvePositionIK:
    """Tests for the solve_position_ik function."""

    # --- Validation tests ---

    def test_invalid_target_position_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="3-element"):
            solve_position_ik(rr, [0.5, 0.3])

    def test_invalid_method_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="Unknown method"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], method="invalid")

    def test_custom_joint_limits_validates_length(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="DOF"):
            solve_position_ik(rr, [1.0, 0.0, 0.0],
                              joint_limits=[(-np.pi, np.pi)])

    # --- CCD tests ---

    def test_ccd_rr_planar_known_solution(self):
        """
        CCD: For a 2R planar robot with l1=1, l2=1:
        q = [pi/3, pi/6] → target → solve IK → should recover q.
        """
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]

        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(rr, target, q0=[0.5, 0.5],
                                method="ccd", tol=1e-8, max_iter=1000)

        assert sol.converged is True
        assert sol.method == "ccd"
        assert_reaches_target(rr, sol, target)

    def test_ccd_rr_planar_reaches_target(self):
        """CCD: Verify that fkine(sol) ≈ target."""
        rr = Robot((1.5, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        target = [2.0, 0.5, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.2, 0.3],
                                method="ccd", tol=1e-8, max_iter=1000)

        assert sol.converged is True
        assert sol.method == "ccd"

        T_sol = rr.T.subs({q1: sol.q[0], q2: sol.q[1]})
        pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
        np.testing.assert_allclose(pos_sol, target, atol=1e-6)

    def test_ccd_out_of_reach_does_not_converge(self):
        """CCD: A target far outside the workspace should not converge."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [10.0, 10.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.0, 0.0],
                                method="ccd", max_iter=200, tol=1e-6)

        assert sol.converged is False

    def test_ccd_respects_joint_limits(self):
        """CCD: Joint limits should be respected."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        limits = [(0, np.pi / 2), (-np.pi / 4, np.pi / 4)]
        target = [1.2, 0.8, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.3, 0.1],
                                method="ccd", joint_limits=limits,
                                tol=1e-8, max_iter=1000)

        assert_within_joint_limits(sol.q, limits)

    def test_ccd_random_initial_guess(self, monkeypatch):
        """CCD: Without providing q0, should still find a solution."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.5, 0.0, 0.0]

        monkeypatch.setattr(
            np.random,
            "uniform",
            lambda lower, upper: np.array([0.4, -0.8], dtype=float),
        )

        sol = solve_position_ik(
            rr, target, method="ccd", tol=1e-6, max_iter=1000
        )

        assert sol.converged is True
        assert_reaches_target(rr, sol, target, atol=1e-5)

    def test_ccd_near_singularity(self):
        """CCD should converge to a target near a singular configuration."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.99, 0.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[1.0, 0.0],
                                method="ccd", tol=1e-8, max_iter=2000)

        assert sol.converged is True

    def test_ccd_prismatic_joint(self):
        """
        CCD: Robot with a prismatic joint.
        1-DOF prismatic: position should match target exactly.
        """
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 3.0]

        sol = solve_position_ik(robot, target, q0=[1.0],
                                method="ccd", tol=1e-8, max_iter=500)

        assert sol.converged is True

        T_sol = robot.T.subs({q1: sol.q[0]})
        pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
        np.testing.assert_allclose(pos_sol, target, atol=1e-6)

    def test_ccd_default_max_iter(self):
        """CCD should perform 500 sweeps by default when it cannot converge."""
        robot = Robot((0, 0, q1, 0, "p"))

        sol = solve_position_ik(
            robot,
            [0.0, 0.0, 10.0],
            q0=[0.0],
            method="ccd",
            joint_limits=[(0.0, 0.1)],
            tol=1e-12,
        )

        assert sol.converged is False
        assert sol.method == "ccd"
        assert sol.iterations == 500

    # --- Levenberg-Marquardt tests ---

    def test_lm_rr_planar_known_solution(self):
        """LM: Known solution for 2R planar robot."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]

        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(rr, target, q0=[0.5, 0.5], tol=1e-8)

        assert sol.converged is True
        assert sol.method == "lm"
        assert_reaches_target(rr, sol, target)

    def test_lm_rr_planar_reaches_target(self):
        """LM: Verify that fkine(sol) ≈ target."""
        rr = Robot((1.5, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        target = [2.0, 0.5, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.2, 0.3], tol=1e-8)

        assert sol.converged is True
        assert sol.method == "lm"

        T_sol = rr.T.subs({q1: sol.q[0], q2: sol.q[1]})
        pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
        np.testing.assert_allclose(pos_sol, target, atol=1e-6)

    def test_lm_out_of_reach_does_not_converge(self):
        """LM: A target far outside the workspace should not converge."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [10.0, 10.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.0, 0.0], max_iter=50, tol=1e-6)

        assert sol.converged is False
        assert sol.method == "lm"

    def test_lm_respects_joint_limits(self):
        """LM: Joint limits should be respected."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        limits = [(0, np.pi / 2), (-np.pi / 4, np.pi / 4)]
        target = [1.5, 0.5, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.3, 0.1],
                                joint_limits=limits, tol=1e-8)

        assert_within_joint_limits(sol.q, limits)

    def test_lm_random_initial_guess(self, monkeypatch):
        """LM: Without providing q0, should still find a solution."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.5, 0.0, 0.0]

        monkeypatch.setattr(
            np.random,
            "uniform",
            lambda lower, upper: np.array([0.4, -0.8], dtype=float),
        )

        sol = solve_position_ik(rr, target, tol=1e-6, max_iter=100)

        assert sol.converged is True
        assert_reaches_target(rr, sol, target, atol=1e-5)

    def test_lm_custom_damping(self):
        """LM: Custom damping parameter should not break the solver."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.2, 0.8, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.1, 0.1],
                                damping=2.0, damping_scale=0.3, tol=1e-8)

        assert sol.converged is True

    def test_lm_near_singularity(self):
        """LM should converge to a target near a singular configuration."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.99, 0.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[1.0, 0.0],
                                damping=10.0, tol=1e-8, max_iter=200)

        assert sol.converged is True

    # --- Newton-Raphson tests ---

    def test_newton_rr_planar_known_solution(self):
        """Newton: Known solution for 2R planar robot."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]

        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(rr, target, q0=[0.5, 0.5],
                                method="newton", tol=1e-8)

        assert sol.converged is True
        assert sol.method == "newton"
        assert_reaches_target(rr, sol, target)

    def test_newton_rr_planar_reaches_target(self):
        """Newton: Verify that fkine(sol) ≈ target."""
        rr = Robot((1.5, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        target = [2.0, 0.5, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.2, 0.3],
                                method="newton", tol=1e-8)

        assert sol.converged is True
        assert sol.method == "newton"

        T_sol = rr.T.subs({q1: sol.q[0], q2: sol.q[1]})
        pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
        np.testing.assert_allclose(pos_sol, target, atol=1e-6)

    def test_newton_out_of_reach_does_not_converge(self):
        """Newton: A target far outside the workspace should not converge."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [10.0, 10.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.0, 0.0],
                                method="newton", max_iter=50, tol=1e-6)

        assert sol.converged is False

    # --- Additional validations requested for robustness ---

    def test_target_position_with_nan_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="target_position"):
            solve_position_ik(rr, [np.nan, 0.0, 0.0], q0=[0.1, 0.1])

    def test_target_position_with_inf_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="target_position"):
            solve_position_ik(rr, [np.inf, 0.0, 0.0], q0=[0.1, 0.1])

    def test_target_position_with_non_numeric_value_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="target_position"):
            solve_position_ik(rr, ["x", 0.0, 0.0], q0=[0.1, 0.1])

    def test_q0_with_non_numeric_value_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="q0"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[0.1, "x"])

    @pytest.mark.parametrize("method", ["newton", "lm", "ccd"])
    def test_symbolic_parameters_are_supported(self, method):
        ls1, ls2 = sp.symbols("ls1 ls2")
        robot = Robot((ls1, 0, 0, q1, "r"), (ls2, 0, 0, q2, "r"))
        T_original = robot.T

        q_known = [np.pi / 3, np.pi / 6]
        substitutions = {ls1: 1.0, ls2: 1.0, q1: q_known[0], q2: q_known[1]}
        T_known = robot.T.subs(substitutions)
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(
            robot,
            target,
            q0=[0.2, 0.2],
            method=method,
            parameters={ls1: 1.0, ls2: 1.0},
            tol=1e-8,
            max_iter=1000 if method == "ccd" else 200,
        )

        assert sol.converged is True

        assert_reaches_target(
            robot, sol, target, parameters={ls1: 1.0, ls2: 1.0}
        )

        # Ensure symbolic expressions on the robot object remain unchanged.
        assert robot.T == T_original
        assert ls1 in robot.T.free_symbols
        assert ls2 in robot.T.free_symbols

    def test_missing_symbolic_parameters_raise_clear_error(self):
        ls1, ls2 = sp.symbols("ls1 ls2")
        robot = Robot((ls1, 0, 0, q1, "r"), (ls2, 0, 0, q2, "r"))

        with pytest.raises(ValueError, match="no numerical value"):
            solve_position_ik(
                robot,
                [1.0, 0.0, 0.0],
                q0=[0.1, 0.1],
                parameters={ls1: 1.0},
            )

    def test_q0_shorter_than_dof_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="Expected size 2, got 1"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[0.1])

    def test_q0_longer_than_dof_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="Expected size 2, got 3"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[0.1, 0.2, 0.3])

    def test_q0_with_nan_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="q0"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[np.nan, 0.1])

    def test_q0_with_inf_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="q0"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[np.inf, 0.1])

    def test_q0_is_clipped_to_joint_limits(self):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        limits = [(-0.2, 0.2), (-0.1, 0.1)]
        q_clipped = [0.2, -0.1]

        T_target = rr.T.subs({q1: q_clipped[0], q2: q_clipped[1]})
        target = [float(T_target[0, 3]), float(T_target[1, 3]), float(T_target[2, 3])]

        sol = solve_position_ik(
            rr,
            target,
            q0=[10.0, -10.0],
            joint_limits=limits,
            method="lm",
            tol=1e-8,
        )

        assert sol.converged is True
        assert sol.iterations == 0
        np.testing.assert_allclose(sol.q, q_clipped, atol=1e-12)

    def test_joint_limits_wrong_number_of_pairs(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="Expected 2, got 1"):
            solve_position_ik(rr, [1.5, 0.2, 0.0], q0=[0.1, 0.1], joint_limits=[(-1, 1)])

    def test_joint_limit_pair_with_wrong_length(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="exactly two elements"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), (-2, 2, 3)],
            )

    def test_joint_limit_pair_must_be_a_sequence(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="sequence of two elements"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), 2],
            )

    def test_joint_limit_with_non_numeric_value(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="real numeric"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), ("a", 2)],
            )

    def test_joint_limit_with_nan_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="finite"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), (np.nan, 2)],
            )

    def test_joint_limit_with_inf_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="finite"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), (-2, np.inf)],
            )

    def test_joint_limit_lower_greater_than_upper(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="greater than"):
            solve_position_ik(
                rr,
                [1.5, 0.2, 0.0],
                q0=[0.1, 0.1],
                joint_limits=[(-1, 1), (2, -2)],
            )

    def test_joint_limit_lower_equal_upper_is_allowed(self):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        limits = [(0.0, 0.0), (-np.pi, np.pi)]

        T_target = rr.T.subs({q1: 0.0, q2: 0.5})
        target = [float(T_target[0, 3]), float(T_target[1, 3]), float(T_target[2, 3])]

        sol = solve_position_ik(
            rr,
            target,
            q0=[0.0, 0.2],
            joint_limits=limits,
            method="lm",
            tol=1e-8,
        )

        assert sol.converged is True
        assert abs(sol.q[0]) <= 1e-12

    def test_robot_joint_limits_are_used_when_joint_limits_is_none(self):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        rr.joint_limits = [(0.0, 0.0), (-np.pi, np.pi)]

        T_target = rr.T.subs({q1: 0.0, q2: 0.4})
        target = [float(T_target[0, 3]), float(T_target[1, 3]), float(T_target[2, 3])]

        sol = solve_position_ik(rr, target, q0=[1.0, 0.0], method="newton", tol=1e-8)
        assert abs(sol.q[0]) <= 1e-12

    @pytest.mark.parametrize("bad_tol", [0.0, -1e-6, np.nan, np.inf])
    def test_invalid_tol_raises_error(self, bad_tol):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="tol"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], tol=bad_tol)

    @pytest.mark.parametrize("bad_iter", [0, -1])
    def test_max_iter_non_positive_raises_error(self, bad_iter):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="max_iter"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], max_iter=bad_iter)

    def test_max_iter_non_integer_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="max_iter"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], max_iter=3.5)

    def test_max_iter_bool_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="max_iter"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], max_iter=True)

    @pytest.mark.parametrize("bad_damping", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_damping_raises_error(self, bad_damping):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="damping"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], damping=bad_damping)

    @pytest.mark.parametrize("bad_scale", [0.0, -0.2, 1.0, 1.2, np.nan, np.inf])
    def test_invalid_damping_scale_raises_error(self, bad_scale):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="damping_scale"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], damping_scale=bad_scale)

    def test_non_finite_during_execution_returns_controlled_failure(self):
        class FakeRobot:
            def __init__(self):
                self.dof = 1
                self.qs = [q1]
                self.joint_limits = [(-1.0, 1.0)]
                self.T = sp.Matrix(
                    [
                        [1, 0, 0, q1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                self.J = sp.Matrix([[sp.nan], [0], [0], [0], [0], [1]])

            def joint_type(self, i):
                return "p"

            def r_o(self, i):
                return sp.Matrix([0, 0, 0])

            def z(self, i):
                return sp.Matrix([0, 0, 1])

        robot = FakeRobot()
        sol = solve_position_ik(robot, [0.5, 0.0, 0.0], q0=[0.0], method="newton", max_iter=5)

        assert sol.converged is False
        assert np.isfinite(sol.q[0])
        assert not np.isfinite(sol.error)
        assert sol.iterations == 0

    def test_iterations_are_zero_when_q0_already_converged(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 0.5]
        sol = solve_position_ik(robot, target, q0=[0.5], method="newton", tol=1e-12)

        assert sol.converged is True
        assert sol.iterations == 0

    def test_iterations_are_one_after_single_update(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 0.5]
        sol = solve_position_ik(robot, target, q0=[0.0], method="newton", tol=1e-12)

        assert sol.converged is True
        assert sol.iterations == 1

    def test_iterations_equal_max_iter_when_not_converged(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 10.0]
        sol = solve_position_ik(
            robot,
            target,
            q0=[0.0],
            method="newton",
            joint_limits=[(0.0, 0.1)],
            max_iter=7,
            tol=1e-12,
        )

        assert sol.converged is False
        assert sol.iterations == 7

    @pytest.mark.parametrize("method", ["newton", "lm"])
    def test_jacobian_methods_default_max_iter(self, method):
        robot = Robot((0, 0, q1, 0, "p"))
        sol = solve_position_ik(
            robot,
            [0.0, 0.0, 10.0],
            q0=[0.0],
            method=method,
            joint_limits=[(0.0, 0.1)],
            tol=1e-12,
        )

        assert sol.converged is False
        assert sol.iterations == 100

    def test_ccd_rejects_unsupported_joint_type(self):
        robot = Robot((0, 0, q1, 0, "p"))
        robot.joint_types[0] = "x"

        with pytest.raises(ValueError, match="Unsupported joint type"):
            solve_position_ik(
                robot,
                [0.0, 0.0, 0.5],
                q0=[0.0],
                method="ccd",
                max_iter=5,
            )

    def test_ccd_prismatic_recomputes_error_within_sweep(self):
        robot = Robot(
            (0, -sp.pi / 2, q1, 0, "p"),
            (1, 0, 0, q2, "r"),
        )

        q_goal = [0.6, 0.4]
        T_goal = robot.T.subs({q1: q_goal[0], q2: q_goal[1]})
        target = [float(T_goal[0, 3]), float(T_goal[1, 3]), float(T_goal[2, 3])]

        sol = solve_position_ik(
            robot,
            target,
            q0=[0.0, 0.0],
            method="ccd",
            max_iter=300,
            tol=1e-8,
            joint_limits=[(-2, 2), (-np.pi, np.pi)],
        )

        assert sol.method == "ccd"
        assert sol.converged is True

        T_sol = robot.T.subs({q1: sol.q[0], q2: sol.q[1]})
        pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
        err = np.linalg.norm(np.asarray(target) - np.asarray(pos_sol))
        assert err < 1e-6