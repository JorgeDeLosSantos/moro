"""
Tests for the inverse kinematics module.
"""
import pytest
import sympy as sp
import numpy as np
from moro.core import Robot
from moro.inverse_kinematics import solve_position_ik, IKSolution
from moro.abc import l1, l2, q1, q2


class TestIKSolution:
    """Tests for the IKSolution data class."""

    def test_creation_and_attributes(self):
        sol = IKSolution([0.5, 1.2], converged=True, iterations=5, error=1e-8)
        assert sol.q == [0.5, 1.2]
        assert sol.converged is True
        assert sol.iterations == 5
        assert sol.error == 1e-8
        assert sol.method == "lm"  # default method

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


class TestSolvePositionIK:
    """Tests for the solve_position_ik function."""

    # --- Validation tests ---

    def test_invalid_target_position_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="3-element"):
            solve_position_ik(rr, [0.5, 0.3])  # only 2 elements

    def test_invalid_method_raises_error(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="Unknown method"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], method="invalid")

    def test_custom_joint_limits_validates_length(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="DOF"):
            solve_position_ik(rr, [1.0, 0.0, 0.0],
                              joint_limits=[(-np.pi, np.pi)])

    # --- Levenberg-Marquardt (default) tests ---

    def test_lm_rr_planar_known_solution(self):
        """
        LM: For a 2R planar robot with l1=1, l2=1:
        If q = [pi/3, pi/6], the position is known.
        Solve IK and verify fkine(sol) approx equals target.
        """
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]

        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(rr, target, q0=[0.5, 0.5], tol=1e-8)

        assert sol.converged is True
        assert sol.method == "lm"
        np.testing.assert_allclose(sol.q, q_known, atol=1e-6)

    def test_lm_rr_planar_reaches_target(self):
        """
        LM: Solve IK and verify that forward kinematics at the solution
        produces the target position.
        """
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

        if sol.converged:
            assert limits[0][0] <= sol.q[0] <= limits[0][1]
            assert limits[1][0] <= sol.q[1] <= limits[1][1]

    def test_lm_random_initial_guess(self):
        """LM: Without providing q0, the solver should still find a solution."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.5, 0.0, 0.0]

        any_converged = False
        for _ in range(5):
            sol = solve_position_ik(rr, target, tol=1e-6, max_iter=100)
            if sol.converged:
                any_converged = True
                T_sol = rr.T.subs({q1: sol.q[0], q2: sol.q[1]})
                pos_sol = [float(T_sol[0, 3]), float(T_sol[1, 3]), float(T_sol[2, 3])]
                np.testing.assert_allclose(pos_sol, target, atol=1e-5)
                break

        assert any_converged, (
            "Solver should find a solution with random initial guesses "
            "for a reachable target."
        )

    def test_lm_custom_damping(self):
        """LM: Custom damping parameter should not break the solver."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.2, 0.8, 0.0]

        sol = solve_position_ik(rr, target, q0=[0.1, 0.1],
                                damping=2.0, damping_scale=0.3, tol=1e-8)

        assert sol.converged is True

    def test_lm_near_singularity(self):
        """
        LM should handle configurations near singularities better than NR.
        A 2R robot with q2 ≈ 0 is near a singularity (elbow fully extended).
        """
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        # A target near the reach limit (hard for Newton-Raphson)
        target = [1.99, 0.0, 0.0]

        sol = solve_position_ik(rr, target, q0=[1.0, 0.0],
                                damping=10.0, tol=1e-8, max_iter=200)

        assert sol.converged is True

    # --- Newton-Raphson tests ---

    def test_newton_rr_planar_known_solution(self):
        """
        Newton: For a 2R planar robot with l1=1, l2=1:
        Solve IK with known solution.
        """
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]

        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(rr, target, q0=[0.5, 0.5],
                                method="newton", tol=1e-8)

        assert sol.converged is True
        assert sol.method == "newton"
        np.testing.assert_allclose(sol.q, q_known, atol=1e-6)

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