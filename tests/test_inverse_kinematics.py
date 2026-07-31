"""
Tests for the inverse kinematics module.
"""
import pytest
import sympy as sp
import numpy as np
from moro.core import Robot
import moro.inverse_kinematics as ik_module
from moro.inverse_kinematics import (
    solve_position_ik,
    solve_position_trajectory,
    IKSolution,
    IKTrajectorySolution,
)
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


def assert_solution_residual_consistency(solution):
    """Assert finite residual consistency for converged solutions."""
    assert isinstance(solution.residual, list)
    assert len(solution.residual) == 3
    assert np.all(np.isfinite(solution.residual))
    assert np.isclose(solution.error, np.linalg.norm(solution.residual))


class TestIKSolution:
    """Tests for the IKSolution data class."""

    def test_creation_and_attributes(self):
        sol = IKSolution([0.5, 1.2], converged=True, iterations=5, error=1e-8)
        assert sol.q == [0.5, 1.2]
        assert sol.converged is True
        assert sol.iterations == 5
        assert sol.error == 1e-8
        assert sol.method == "lm"
        assert sol.residual is None
        assert sol.message == ""

    def test_legacy_constructor_without_new_fields(self):
        sol = IKSolution(0.5, True, 1, 0.0)
        assert sol.q == [0.5]
        assert sol.method == "lm"
        assert sol.residual is None
        assert sol.message == ""

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

    def test_iterable_q_converted_to_list(self):
        sol = IKSolution(np.array([0.1, 0.2]), converged=False, iterations=0, error=1.0)
        assert sol.q == [0.1, 0.2]

    def test_numpy_residual_converted_to_list(self):
        sol = IKSolution(
            [0.2],
            converged=False,
            iterations=2,
            error=0.3,
            residual=np.array([0.1, 0.2, 0.0]),
        )
        assert sol.residual == [0.1, 0.2, 0.0]

    def test_custom_method_in_repr(self):
        sol = IKSolution([0.2], converged=True, iterations=4, error=1e-8, method="newton")
        assert "newton" in repr(sol)

    def test_ccd_method_in_repr(self):
        sol = IKSolution([0.2], converged=True, iterations=4, error=1e-8, method="ccd")
        assert "ccd" in repr(sol)

    def test_repr_keeps_main_information(self):
        sol = IKSolution([0.2, -0.1], converged=False, iterations=7, error=2e-2, method="lm")
        text = repr(sol)
        assert "IKSolution" in text
        assert "method=lm" in text
        assert "iters=7" in text
        assert "error=" in text


class TestIKTrajectorySolution:
    """Tests for the IKTrajectorySolution data class."""

    def test_creation_and_attributes(self):
        sol1 = IKSolution([0.1, 0.2], True, 3, 1e-9, residual=[0.0, 0.0, 0.0])
        sol2 = IKSolution([0.2, 0.3], True, 4, 2e-9, residual=[0.0, 0.0, 0.0])

        traj = IKTrajectorySolution(
            solutions=(sol1, sol2),
            converged=1,
            failed_index=None,
            message=123,
        )

        assert isinstance(traj.solutions, list)
        assert len(traj.solutions) == 2
        assert traj.converged is True
        assert traj.failed_index is None
        assert traj.message == "123"

    def test_rejects_non_iksolution_elements(self):
        sol = IKSolution([0.1], True, 1, 0.0, residual=[0.0, 0.0, 0.0])
        with pytest.raises(TypeError, match="solutions\\[1\\]"):
            IKTrajectorySolution(solutions=[sol, "bad"], converged=False)

    def test_failed_index_validation(self):
        sol = IKSolution([0.1], False, 1, 0.1, residual=[0.1, 0.0, 0.0])

        with pytest.raises(ValueError, match="failed_index"):
            IKTrajectorySolution(solutions=[sol], converged=False, failed_index=-1)

        with pytest.raises(ValueError, match="failed_index"):
            IKTrajectorySolution(solutions=[sol], converged=False, failed_index=1.2)

    def test_converged_requires_none_failed_index(self):
        sol = IKSolution([0.1], True, 1, 0.0, residual=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="failed_index"):
            IKTrajectorySolution(solutions=[sol], converged=True, failed_index=0)

    def test_properties_qs_errors_and_iterations(self):
        sol1 = IKSolution([0.1, 0.2], True, 3, 1e-4, residual=[1e-4, 0.0, 0.0])
        sol2 = IKSolution([0.2, 0.4], False, 5, 2e-3, residual=[2e-3, 0.0, 0.0])

        traj = IKTrajectorySolution(
            solutions=[sol1, sol2],
            converged=False,
            failed_index=1,
            message="failed",
        )

        assert traj.qs == [[0.1, 0.2], [0.2, 0.4]]
        assert traj.errors == [1e-4, 2e-3]
        assert traj.iterations == [3, 5]

    def test_qs_returns_copies(self):
        sol = IKSolution([0.3, 0.4], True, 2, 0.0, residual=[0.0, 0.0, 0.0])
        traj = IKTrajectorySolution(solutions=[sol], converged=True)

        qs = traj.qs
        qs[0][0] = -123.0

        assert traj.solutions[0].q == [0.3, 0.4]

    def test_repr_is_compact(self):
        sol = IKSolution([0.1], False, 2, 1e-2, residual=[1e-2, 0.0, 0.0])
        traj = IKTrajectorySolution(
            solutions=[sol],
            converged=False,
            failed_index=0,
            message="failed",
        )

        text = repr(traj)
        assert "IKTrajectorySolution" in text
        assert "points=1" in text
        assert "failed_index=0" in text
        assert "IKSolution(" not in text


class TestSolvePositionTrajectory:
    """Tests for the solve_position_trajectory function."""

    def test_rr_trajectory_reachable(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        q_refs = [[0.2, 0.3], [0.25, 0.35], [0.3, 0.4], [0.35, 0.45]]
        targets = [end_effector_position(rr, q) for q in q_refs]

        trajectory = solve_position_trajectory(
            rr,
            targets,
            q0=[0.1, 0.1],
            method="lm",
            tol=1e-8,
        )

        assert trajectory.converged is True
        assert trajectory.failed_index is None
        assert len(trajectory.solutions) == len(targets)
        assert len(trajectory.qs) == len(targets)

        for solution, target in zip(trajectory.solutions, targets):
            assert isinstance(solution, IKSolution)
            assert_reaches_target(rr, solution, target)

    def test_mixed_pr_trajectory_reachable(self):
        pr = Robot((0, 0, q1, 0, "p"), (1.0, 0, 0, q2, "r"))
        q_refs = [[0.3, 0.2], [0.4, 0.25], [0.5, 0.3], [0.6, 0.35]]
        targets = [end_effector_position(pr, q) for q in q_refs]

        trajectory = solve_position_trajectory(
            pr,
            targets,
            q0=[0.1, 0.1],
            method="ccd",
            tol=1e-8,
            max_iter=1000,
        )

        assert trajectory.converged is True
        assert trajectory.failed_index is None
        assert len(trajectory.solutions) == len(targets)

        for solution, target in zip(trajectory.solutions, targets):
            assert_reaches_target(pr, solution, target, atol=1e-6)

    def test_symbolic_parameters_are_supported(self):
        ls1, ls2 = sp.symbols("ls1 ls2")
        robot = Robot((ls1, 0, 0, q1, "r"), (ls2, 0, 0, q2, "r"))
        original_T = robot.T

        q_refs = [[0.3, 0.4], [0.35, 0.45], [0.4, 0.5]]
        params = {ls1: 1.0, ls2: 1.0}
        targets = [end_effector_position(robot, q, parameters=params) for q in q_refs]

        trajectory = solve_position_trajectory(
            robot,
            targets,
            q0=[0.2, 0.2],
            parameters=params,
            method="newton",
            tol=1e-8,
        )

        assert trajectory.converged is True
        assert trajectory.failed_index is None
        assert len(trajectory.solutions) == len(targets)

        for solution, target in zip(trajectory.solutions, targets):
            assert_reaches_target(robot, solution, target, parameters=params)

        assert robot.T == original_T
        assert ls1 in robot.T.free_symbols
        assert ls2 in robot.T.free_symbols

    def test_reuses_previous_solution_as_seed(self, monkeypatch):
        received_q0 = []

        def fake_solver(
            robot,
            target_position,
            q0=None,
            joint_limits=None,
            tol=1e-6,
            max_iter=None,
            method="lm",
            damping=1.0,
            damping_scale=0.5,
            *,
            parameters=None,
            random_state=None,
            step_tol=1e-12,
            error_change_tol=1e-12,
            stagnation_iterations=5,
        ):
            received_q0.append(list(q0))
            q_next = [q0[0] + 1.0, q0[1] + 2.0]
            return IKSolution(
                q=q_next,
                converged=True,
                iterations=1,
                error=0.0,
                method=method,
                residual=[0.0, 0.0, 0.0],
                message="Converged successfully.",
            )

        monkeypatch.setattr(ik_module, "solve_position_ik", fake_solver)

        trajectory = solve_position_trajectory(
            robot=object(),
            target_positions=[[1.0, 0.0, 0.0], [1.1, 0.1, 0.0], [1.2, 0.2, 0.0]],
            q0=[0.1, 0.2],
        )

        assert trajectory.converged is True
        assert received_q0[0] == [0.1, 0.2]
        assert received_q0[1] == [1.1, 2.2]
        assert received_q0[2] == [2.1, 4.2]

    def test_stops_on_first_failure_and_keeps_failing_solution(self, monkeypatch):
        call_count = {"n": 0}

        def fake_solver(
            robot,
            target_position,
            q0=None,
            joint_limits=None,
            tol=1e-6,
            max_iter=None,
            method="lm",
            damping=1.0,
            damping_scale=0.5,
            *,
            parameters=None,
            random_state=None,
            step_tol=1e-12,
            error_change_tol=1e-12,
            stagnation_iterations=5,
        ):
            idx = call_count["n"]
            call_count["n"] += 1

            if idx == 0:
                return IKSolution(
                    q=[0.3, 0.4],
                    converged=True,
                    iterations=2,
                    error=0.0,
                    method=method,
                    residual=[0.0, 0.0, 0.0],
                    message="Converged successfully.",
                )

            if idx == 1:
                return IKSolution(
                    q=[0.6, 0.7],
                    converged=False,
                    iterations=50,
                    error=0.5,
                    method=method,
                    residual=[0.5, 0.0, 0.0],
                    message="Maximum number of iterations reached.",
                )

            pytest.fail("solve_position_ik should not be called after first failure.")

        monkeypatch.setattr(ik_module, "solve_position_ik", fake_solver)

        trajectory = solve_position_trajectory(
            robot=object(),
            target_positions=[[1.0, 0.0, 0.0], [1.2, 0.2, 0.0], [1.3, 0.3, 0.0]],
            q0=[0.1, 0.2],
        )

        assert trajectory.converged is False
        assert trajectory.failed_index == 1
        assert len(trajectory.solutions) == 2
        assert "target index 1" in trajectory.message
        assert "Maximum number of iterations reached." in trajectory.message

    def test_rejects_empty_target_positions(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match="at least one target"):
            solve_position_trajectory(rr, [], q0=[0.1, 0.1])

    def test_rejects_target_with_two_components(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match=r"target_positions\[1\]"):
            solve_position_trajectory(rr, [[1.0, 0.0, 0.0], [0.5, 0.1]], q0=[0.1, 0.1])

    def test_rejects_target_with_four_components(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match=r"target_positions\[0\]"):
            solve_position_trajectory(rr, [[1.0, 0.0, 0.0, 0.0]], q0=[0.1, 0.1])

    def test_rejects_target_with_nan(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match=r"target_positions\[0\]"):
            solve_position_trajectory(rr, [[np.nan, 0.0, 0.0]], q0=[0.1, 0.1])

    def test_rejects_target_with_inf(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match=r"target_positions\[0\]"):
            solve_position_trajectory(rr, [[np.inf, 0.0, 0.0]], q0=[0.1, 0.1])

    def test_rejects_target_with_non_numeric_value(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match=r"target_positions\[0\]"):
            solve_position_trajectory(rr, [["x", 0.0, 0.0]], q0=[0.1, 0.1])

    def test_rejects_single_vector_numpy_array(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        with pytest.raises(ValueError, match="single 3-element vector"):
            solve_position_trajectory(rr, np.array([1.0, 0.0, 0.0]), q0=[0.1, 0.1])

    def test_accepts_matrix_numpy_array(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        q_refs = [[0.2, 0.3], [0.25, 0.35], [0.3, 0.4]]
        targets = np.asarray([end_effector_position(rr, q) for q in q_refs], dtype=float)

        trajectory = solve_position_trajectory(rr, targets, q0=[0.1, 0.1], tol=1e-8)

        assert trajectory.converged is True
        assert len(trajectory.solutions) == targets.shape[0]

    def test_integration_with_iksolution_outputs(self):
        rr = Robot((1.0, 0, 0, q1, "r"), (1.0, 0, 0, q2, "r"))
        q_refs = [[0.2, 0.4], [0.3, 0.5], [0.35, 0.55]]
        targets = [end_effector_position(rr, q) for q in q_refs]

        trajectory = solve_position_trajectory(rr, targets, q0=[0.1, 0.1], tol=1e-8)

        assert trajectory.converged is True
        assert all(isinstance(solution, IKSolution) for solution in trajectory.solutions)
        assert trajectory.errors == [solution.error for solution in trajectory.solutions]
        assert trajectory.qs == [solution.q for solution in trajectory.solutions]


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

    @pytest.mark.parametrize("method", ["newton", "lm", "ccd"])
    def test_converged_solution_exposes_residual_and_message(self, method):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        q_known = [np.pi / 3, np.pi / 6]
        T_known = rr.T.subs({q1: q_known[0], q2: q_known[1]})
        target = [float(T_known[0, 3]), float(T_known[1, 3]), float(T_known[2, 3])]

        sol = solve_position_ik(
            rr,
            target,
            q0=[0.4, 0.4],
            method=method,
            tol=1e-8,
            max_iter=1000 if method == "ccd" else 200,
        )

        assert sol.converged is True
        assert sol.message == "Converged successfully."
        assert_solution_residual_consistency(sol)

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

    def test_ccd_random_initial_guess(self):
        """CCD: Without providing q0, should still find a solution."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.5, 0.0, 0.0]

        sol = solve_position_ik(
            rr, target, method="ccd", tol=1e-6, max_iter=1000, random_state=42
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
            step_tol=0.0,
            error_change_tol=0.0,
            stagnation_iterations=501,
        )

        assert sol.converged is False
        assert sol.method == "ccd"
        assert sol.iterations == 500
        assert sol.message == "Maximum number of iterations reached."

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

    def test_lm_random_initial_guess(self):
        """LM: Without providing q0, should still find a solution."""
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.5, 0.0, 0.0]

        sol = solve_position_ik(rr, target, tol=1e-6, max_iter=100, random_state=42)

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
            q0=[0.8, 0.8],
            method=method,
            parameters={ls1: 1.0, ls2: 1.0},
            tol=1e-8,
            max_iter=1000 if method == "ccd" else 200,
            step_tol=0.0,
            error_change_tol=0.0,
            stagnation_iterations=1001 if method == "ccd" else 201,
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

    @pytest.mark.parametrize("bad_step_tol", [-1e-9, np.nan, np.inf, True])
    def test_invalid_step_tol_raises_error(self, bad_step_tol):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="step_tol"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], q0=[0.1, 0.1], step_tol=bad_step_tol)

    @pytest.mark.parametrize("bad_error_change_tol", [-1e-9, np.nan, np.inf])
    def test_invalid_error_change_tol_raises_error(self, bad_error_change_tol):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="error_change_tol"):
            solve_position_ik(
                rr,
                [1.0, 0.0, 0.0],
                q0=[0.1, 0.1],
                error_change_tol=bad_error_change_tol,
            )

    @pytest.mark.parametrize("bad_stagnation_iterations", [0, 2.5, True])
    def test_invalid_stagnation_iterations_raises_error(self, bad_stagnation_iterations):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="stagnation_iterations"):
            solve_position_ik(
                rr,
                [1.0, 0.0, 0.0],
                q0=[0.1, 0.1],
                stagnation_iterations=bad_stagnation_iterations,
            )

    def test_random_state_bool_is_rejected(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises(ValueError, match="random_state"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], random_state=True)

    def test_random_state_invalid_type_is_rejected(self):
        rr = Robot((1, 0, 0, q1), (1, 0, 0, q2))
        with pytest.raises((TypeError, ValueError), match="random_state"):
            solve_position_ik(rr, [1.0, 0.0, 0.0], random_state="42")

    def test_newton_stagnates_when_joint_update_is_too_small(self):
        robot = Robot((0, 0, q1, 0, "p"))
        sol = solve_position_ik(
            robot,
            [0.0, 0.0, 1.0],
            q0=[0.0],
            method="newton",
            joint_limits=[(0.0, 0.0)],
            max_iter=50,
            tol=1e-12,
            stagnation_iterations=1,
        )

        assert sol.converged is False
        assert sol.iterations < 50
        assert sol.message == "Solver stagnated because the joint update became too small."

    def test_lm_stagnates_when_joint_update_is_too_small(self):
        robot = Robot((0, 0, q1, 0, "p"))
        sol = solve_position_ik(
            robot,
            [0.0, 0.0, 1.0],
            q0=[0.0],
            method="lm",
            joint_limits=[(0.0, 0.0)],
            max_iter=50,
            tol=1e-12,
            stagnation_iterations=2,
        )

        assert sol.converged is False
        assert sol.iterations < 50
        assert sol.message == "Solver stagnated because the joint update became too small."

    def test_ccd_stagnates_when_joint_update_is_too_small(self):
        robot = Robot((0, 0, q1, 0, "p"))
        sol = solve_position_ik(
            robot,
            [0.0, 0.0, 1.0],
            q0=[0.0],
            method="ccd",
            joint_limits=[(0.0, 0.0)],
            max_iter=50,
            tol=1e-12,
            stagnation_iterations=1,
        )

        assert sol.converged is False
        assert sol.iterations < 50
        assert sol.message == "Solver stagnated because the joint update became too small."

    def test_stagnates_when_error_stops_improving(self):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        sol = solve_position_ik(
            rr,
            [10.0, 10.0, 0.0],
            q0=[0.0, 0.0],
            method="newton",
            tol=1e-12,
            max_iter=100,
            step_tol=1e-20,
            error_change_tol=1e6,
            stagnation_iterations=2,
        )

        assert sol.converged is False
        assert sol.iterations == 2
        assert sol.message == "Solver stagnated because the position error stopped improving."

    def test_same_integer_seed_is_reproducible_when_q0_is_none(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [1.0, 0.0, 0.0]

        sol1 = solve_position_ik(robot, target, method="newton", max_iter=1, random_state=42)
        sol2 = solve_position_ik(robot, target, method="newton", max_iter=1, random_state=42)

        np.testing.assert_allclose(sol1.q, sol2.q, atol=0.0)
        assert sol1.converged == sol2.converged
        assert sol1.iterations == sol2.iterations
        assert sol1.message == sol2.message

    def test_different_integer_seeds_change_initialization(self, monkeypatch):
        rr = Robot((1, 0, 0, q1, "r"), (1, 0, 0, q2, "r"))
        target = [1.0, 0.0, 0.0]
        captured = []

        def capture_solver(
            fk_func,
            j_func,
            target,
            q,
            lower_bounds,
            upper_bounds,
            tol,
            max_iter,
            method,
            damping,
            damping_scale,
            step_tol,
            error_change_tol,
            stagnation_iterations,
        ):
            captured.append(np.asarray(q, dtype=float).copy())
            return IKSolution(
                q=q,
                converged=False,
                iterations=0,
                error=np.inf,
                method=method,
                residual=None,
                message="Captured initial guess.",
            )

        monkeypatch.setattr(ik_module, "_solve_newton_or_lm", capture_solver)
        solve_position_ik(rr, target, method="newton", max_iter=1, random_state=1)
        solve_position_ik(rr, target, method="newton", max_iter=1, random_state=2)

        assert len(captured) == 2
        assert not np.allclose(captured[0], captured[1])

    def test_q0_makes_random_state_irrelevant(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 0.5]

        sol1 = solve_position_ik(
            robot,
            target,
            q0=[0.0],
            method="newton",
            random_state=1,
            tol=1e-12,
        )
        sol2 = solve_position_ik(
            robot,
            target,
            q0=[0.0],
            method="newton",
            random_state=2,
            tol=1e-12,
        )

        np.testing.assert_allclose(sol1.q, sol2.q)
        assert sol1.iterations == sol2.iterations
        assert sol1.error == sol2.error

    def test_random_state_accepts_numpy_generator(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [1.0, 0.0, 0.0]
        rng = np.random.default_rng(123)

        sol = solve_position_ik(robot, target, method="newton", max_iter=1, random_state=rng)

        assert isinstance(sol.q, list)

    def test_global_numpy_random_state_is_not_modified(self):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [1.0, 0.0, 0.0]

        np.random.seed(2026)
        expected_next = np.random.random()

        np.random.seed(2026)
        solve_position_ik(robot, target, method="newton", max_iter=1, random_state=42)
        observed_next = np.random.random()

        assert observed_next == expected_next

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
        assert sol.residual is not None
        assert np.isfinite(sol.error)
        assert sol.iterations == 0
        assert sol.message == "Numerical failure while evaluating the Jacobian."

    def test_failure_without_finite_fk_returns_none_residual(self, monkeypatch):
        robot = Robot((0, 0, q1, 0, "p"))
        target = [0.0, 0.0, 1.0]

        original_safe_eval_vector = ik_module._safe_eval_vector
        call_counter = {"count": 0}

        def fail_after_first_fk(func, q, size):
            call_counter["count"] += 1
            if call_counter["count"] >= 2:
                return None
            return original_safe_eval_vector(func, q, size)

        monkeypatch.setattr(ik_module, "_safe_eval_vector", fail_after_first_fk)

        sol = solve_position_ik(robot, target, q0=[0.0], method="newton", max_iter=5)

        assert sol.converged is False
        assert sol.residual is None
        assert not np.isfinite(sol.error)
        assert sol.message == "Numerical failure while evaluating the forward kinematics."

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
            step_tol=0.0,
            error_change_tol=0.0,
            stagnation_iterations=8,
        )

        assert sol.converged is False
        assert sol.iterations == 7
        assert sol.message == "Maximum number of iterations reached."
        assert sol.residual is not None
        assert len(sol.residual) == 3
        assert np.all(np.isfinite(sol.residual))
        assert np.isclose(sol.error, np.linalg.norm(sol.residual))

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
            step_tol=0.0,
            error_change_tol=0.0,
            stagnation_iterations=101,
        )

        assert sol.converged is False
        assert sol.iterations == 100
        assert sol.message == "Maximum number of iterations reached."

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