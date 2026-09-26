"""Tests for moro.differential_kinematics."""

import math

import numpy as np
import pytest
import sympy as sp

from moro.abc import l1, l2, q1, q2
from moro.core import Robot
from moro.differential_kinematics import (
    VelocityIKSolution,
    cartesian_velocity,
    solve_velocity_ik,
    task_jacobian,
)


def assert_matrix_equal(a, b):
    diff = sp.Matrix(a) - sp.Matrix(b)
    assert all(sp.simplify(value) == 0 for value in diff)


def assert_matrix_close(a, b, atol=1e-9):
    aa = np.asarray(sp.Matrix(a), dtype=float)
    bb = np.asarray(sp.Matrix(b), dtype=float)
    np.testing.assert_allclose(aa, bb, atol=atol, rtol=0.0)


class DummyRobot:
    def __init__(self, J, qs):
        self.J = sp.Matrix(J)
        self.qs = list(qs)
        self.dof = len(self.qs)


def dummy_robot(J):
    qs = sp.symbols(f"q0:{sp.Matrix(J).cols}", real=True)
    return DummyRobot(J, qs)


def test_task_jacobian_uses_robot_geometric_jacobian():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))

    result = task_jacobian(robot)

    assert_matrix_equal(result, robot.J)


@pytest.mark.parametrize(
    "task,rows",
    [
        ("linear", [0, 1, 2]),
        ("angular", [3, 4, 5]),
        ("twist", [0, 1, 2, 3, 4, 5]),
        (("vx", "wz"), [0, 5]),
        (("wz", "vx"), [5, 0]),
        (("VX", "WZ"), [0, 5]),
    ],
)
def test_task_jacobian_selects_rows_and_preserves_order(task, rows):
    J = sp.Matrix(6, 2, range(12))
    robot = DummyRobot(J, sp.symbols("q0:2"))

    result = task_jacobian(robot, task=task)

    assert_matrix_equal(result, J[rows, :])


def test_task_jacobian_substitutes_configuration_then_parameters():
    qa, qb = sp.symbols("qa qb", real=True)
    length = sp.symbols("length", real=True)
    J = sp.Matrix([
        [qa + length, qb],
        [length * qb, qa],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
    ])
    robot = DummyRobot(J, [qa, qb])

    result = task_jacobian(
        robot,
        q=[sp.Rational(1, 2), 2],
        task=("vx", "vy"),
        parameters={length: 3},
    )

    expected = sp.Matrix([
        [sp.Rational(7, 2), 2],
        [6, sp.Rational(1, 2)],
    ])
    assert_matrix_equal(result, expected)


def test_task_jacobian_allows_unresolved_model_parameters():
    robot = Robot((l1, 0, 0, q1),)

    result = task_jacobian(robot, q=[0], task="linear")

    assert l1 in result.free_symbols


@pytest.mark.parametrize("task", ["unknown", "VXWZ", ""])
def test_task_jacobian_rejects_unknown_preset(task):
    robot = dummy_robot(sp.zeros(6, 1))
    with pytest.raises(ValueError):
        task_jacobian(robot, task=task)


@pytest.mark.parametrize(
    "task",
    [
        (),
        ("vx", "vx"),
        ("vx", "bad"),
    ],
)
def test_task_jacobian_rejects_invalid_explicit_task(task):
    robot = dummy_robot(sp.zeros(6, 1))
    with pytest.raises(ValueError):
        task_jacobian(robot, task=task)


@pytest.mark.parametrize("task", [1, None, {"vx"}, ["vx", 1]])
def test_task_jacobian_rejects_invalid_task_types(task):
    robot = dummy_robot(sp.zeros(6, 1))
    with pytest.raises(TypeError):
        task_jacobian(robot, task=task)


def test_task_jacobian_rejects_wrong_configuration_size():
    robot = dummy_robot(sp.zeros(6, 2))
    with pytest.raises(ValueError, match="q must contain exactly 2"):
        task_jacobian(robot, q=[0], task="linear")


def test_task_jacobian_rejects_none_inside_configuration():
    robot = dummy_robot(sp.zeros(6, 2))
    with pytest.raises(ValueError, match="must not contain None"):
        task_jacobian(robot, q=[0, None], task="linear")


def test_task_jacobian_rejects_string_parameter_keys():
    robot = dummy_robot(sp.zeros(6, 1))
    with pytest.raises(TypeError, match="SymPy objects"):
        task_jacobian(robot, q=[0], parameters={"length": 1})


def test_task_jacobian_reports_missing_robot_interface():
    class BadRobot:
        dof = 1

    with pytest.raises(TypeError, match="missing"):
        task_jacobian(BadRobot())


def test_cartesian_velocity_matches_jacobian_product_for_planar_rr():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    q = [sp.pi / 6, sp.pi / 3]
    qd = [2, -1]

    result = cartesian_velocity(
        robot,
        q,
        qd,
        task=("vx", "vy", "wz"),
    )
    expected = task_jacobian(
        robot,
        q,
        task=("vx", "vy", "wz"),
    ) * sp.Matrix(qd)

    assert_matrix_equal(result, expected)


def test_cartesian_velocity_preserves_exact_sympy_arithmetic():
    robot = dummy_robot(
        sp.Matrix([
            [1, 2],
            [3, 4],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ])
    )

    result = cartesian_velocity(
        robot,
        [0, 0],
        [sp.Rational(1, 2), sp.Rational(1, 3)],
        task=("vx", "vy"),
    )

    assert_matrix_equal(
        result,
        sp.Matrix([
            sp.Rational(7, 6),
            sp.Rational(17, 6),
        ]),
    )


def test_cartesian_velocity_parameter_substitution_can_resolve_qd():
    speed = sp.symbols("speed", real=True)
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    result = cartesian_velocity(
        robot,
        [0],
        [speed],
        task=("vx",),
        parameters={speed: sp.Rational(3, 2)},
    )

    assert_matrix_equal(result, sp.Matrix([sp.Rational(3, 2)]))


def test_cartesian_velocity_rejects_unresolved_symbols():
    speed = sp.symbols("speed", real=True)
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises(ValueError, match="unresolved symbols"):
        cartesian_velocity(
            robot,
            [0],
            [speed],
            task=("vx",),
        )


def test_cartesian_velocity_rejects_scalar_even_for_one_dof():
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises((TypeError, ValueError)):
        cartesian_velocity(robot, [0], 2.0, task=("vx",))


def test_velocity_ik_pinv_square_full_rank():
    robot = dummy_robot(
        sp.Matrix([
            [2, 0],
            [0, 4],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [2, 8],
        task=("vx", "vy"),
    )

    assert isinstance(solution, VelocityIKSolution)
    assert_matrix_close(solution.qd, [1, 2])
    assert_matrix_close(solution.achieved_velocity, [2, 8])
    assert_matrix_close(solution.residual, [0, 0])
    assert solution.residual_norm <= 1e-12
    assert solution.rank == 2
    assert solution.condition_number == pytest.approx(2.0)
    assert solution.method == "pinv"
    assert solution.success is True
    assert solution.limited is False


def test_velocity_ik_redundant_returns_minimum_norm_solution():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0, 0],
        [1, 2],
        task=("vx", "vy"),
    )

    assert_matrix_close(solution.qd, [1, 2, 0])
    assert solution.success is True
    assert solution.rank == 2
    assert solution.condition_number == pytest.approx(1.0)


def test_velocity_ik_overdetermined_reachable_task():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [1, 2, 3],
        task="linear",
    )

    assert_matrix_close(solution.qd, [1, 2])
    assert solution.success is True
    assert solution.residual_norm <= 1e-12


def test_velocity_ik_overdetermined_unreachable_task_is_least_squares():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [1, 2, 4],
        task="linear",
        tol=1e-12,
    )

    assert solution.success is False
    assert solution.residual_norm > 0
    assert_matrix_close(
        solution.qd,
        np.linalg.pinv(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
        @ np.array([1, 2, 4], dtype=float),
    )


def test_velocity_ik_rank_deficient_but_attainable_can_succeed():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [2, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [1, 2],
        task=("vx", "vy"),
    )

    assert solution.rank == 1
    assert math.isinf(solution.condition_number)
    assert solution.success is True
    assert_matrix_close(solution.qd, [1, 0])


def test_velocity_ik_zero_jacobian_zero_velocity_succeeds():
    robot = dummy_robot(sp.zeros(6, 2))

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [0, 0],
        task=("vx", "vy"),
    )

    assert_matrix_close(solution.qd, [0, 0])
    assert solution.rank == 0
    assert math.isinf(solution.condition_number)
    assert solution.success is True


def test_velocity_ik_zero_jacobian_nonzero_velocity_fails():
    robot = dummy_robot(sp.zeros(6, 2))

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [1, 0],
        task=("vx", "vy"),
    )

    assert_matrix_close(solution.qd, [0, 0])
    assert solution.rank == 0
    assert math.isinf(solution.condition_number)
    assert solution.success is False
    assert solution.residual_norm == pytest.approx(1.0)


def test_velocity_ik_dls_matches_svd_formula():
    robot = dummy_robot(
        sp.Matrix([
            [2, 0],
            [0, 0.5],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )
    damping = 0.25
    desired = np.array([1.0, 2.0])

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        desired,
        task=("vx", "vy"),
        method="dls",
        damping=damping,
        tol=1.0,
    )

    singular_values = np.array([2.0, 0.5])
    expected = (
        singular_values / (singular_values**2 + damping**2)
    ) * desired

    assert_matrix_close(solution.qd, expected)
    assert solution.method == "dls"
    assert solution.rank == 2


def test_velocity_ik_dls_regularizes_near_singular_joint_velocity():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [0, 1e-8],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    pinv = solve_velocity_ik(
        robot,
        [0, 0],
        [0, 1],
        task=("vx", "vy"),
        method="pinv",
    )
    dls = solve_velocity_ik(
        robot,
        [0, 0],
        [0, 1],
        task=("vx", "vy"),
        method="dls",
        damping=1e-3,
        tol=1.0,
    )

    assert abs(float(dls.qd[1])) < abs(float(pinv.qd[1]))
    assert dls.condition_number == pytest.approx(pinv.condition_number)


@pytest.mark.parametrize(
    "method,damping,exception",
    [
        ("bad", None, ValueError),
        (1, None, TypeError),
        ("pinv", 0.1, ValueError),
        ("dls", None, ValueError),
        ("dls", 0, ValueError),
        ("dls", -1, ValueError),
        ("dls", np.inf, ValueError),
    ],
)
def test_velocity_ik_validates_method_and_damping(method, damping, exception):
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises(exception):
        solve_velocity_ik(
            robot,
            [0],
            [1],
            task=("vx",),
            method=method,
            damping=damping,
        )


@pytest.mark.parametrize("tol", [0, -1, np.inf, np.nan])
def test_velocity_ik_rejects_invalid_tolerance(tol):
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises(ValueError):
        solve_velocity_ik(
            robot,
            [0],
            [1],
            task=("vx",),
            tol=tol,
        )


def test_velocity_ik_rejects_boolean_tolerance():
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises(TypeError):
        solve_velocity_ik(
            robot,
            [0],
            [1],
            task=("vx",),
            tol=True,
        )


def test_velocity_ik_requires_velocity_matching_task_dimension():
    robot = dummy_robot(sp.zeros(6, 2))

    with pytest.raises(ValueError, match="velocity must contain exactly 2"):
        solve_velocity_ik(
            robot,
            [0, 0],
            [1],
            task=("vx", "vy"),
        )


def test_velocity_ik_rejects_unresolved_jacobian_parameters():
    length = sp.symbols("length", real=True)
    robot = dummy_robot(
        sp.Matrix([
            [length],
            [0],
            [0],
            [0],
            [0],
            [0],
        ])
    )

    with pytest.raises(ValueError, match="unresolved symbols"):
        solve_velocity_ik(
            robot,
            [0],
            [1],
            task=("vx",),
        )


def test_velocity_ik_rejects_nonfinite_desired_velocity():
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))

    with pytest.raises(ValueError, match="finite"):
        solve_velocity_ik(
            robot,
            [0],
            [sp.oo],
            task=("vx",),
        )


def test_velocity_ik_symmetric_limits_clip_and_recompute_diagnostics():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [2, -3],
        task=("vx", "vy"),
        joint_velocity_limits=[1.0, 0.5],
        tol=1e-12,
    )

    assert_matrix_close(solution.unconstrained_qd, [2, -3])
    assert_matrix_close(solution.qd, [1, -0.5])
    assert_matrix_close(solution.achieved_velocity, [1, -0.5])
    assert_matrix_close(solution.residual, [1, -2.5])
    assert solution.limited is True
    assert solution.success is False


def test_velocity_ik_asymmetric_limits_clip_independently():
    robot = dummy_robot(
        sp.Matrix([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0, 0],
        [-2, 3],
        task=("vx", "vy"),
        joint_velocity_limits=[(-1.5, 0.5), (-2.0, 2.5)],
    )

    assert_matrix_close(solution.qd, [-1.5, 2.5])
    assert solution.limited is True


@pytest.mark.parametrize(
    "limits,exception",
    [
        ([1.0], ValueError),
        ([1.0, None], ValueError),
        ([0.0, 1.0], ValueError),
        ([-1.0, 1.0], ValueError),
        ([(1.0, 1.0), (-1.0, 1.0)], ValueError),
        ([(2.0, 1.0), (-1.0, 1.0)], ValueError),
        ([(0.0, np.inf), (-1.0, 1.0)], ValueError),
        ("bad", TypeError),
    ],
)
def test_velocity_ik_validates_joint_velocity_limits(limits, exception):
    robot = dummy_robot(sp.zeros(6, 2))

    with pytest.raises(exception):
        solve_velocity_ik(
            robot,
            [0, 0],
            [0, 0],
            task=("vx", "vy"),
            joint_velocity_limits=limits,
        )


def test_velocity_ik_task_order_defines_velocity_order():
    robot = dummy_robot(
        sp.Matrix([
            [2],
            [0],
            [0],
            [0],
            [0],
            [3],
        ])
    )

    solution = solve_velocity_ik(
        robot,
        [0],
        [6, 4],
        task=("wz", "vx"),
        tol=1.0,
    )

    expected = np.linalg.pinv(np.array([[3.0], [2.0]])) @ np.array([6.0, 4.0])
    assert_matrix_close(solution.qd, expected)
    assert_matrix_close(solution.desired_velocity, [6, 4])


def test_velocity_ik_solution_is_frozen():
    robot = dummy_robot(sp.Matrix([[1], [0], [0], [0], [0], [0]]))
    solution = solve_velocity_ik(robot, [0], [1], task=("vx",))

    with pytest.raises(Exception):
        solution.rank = 2
