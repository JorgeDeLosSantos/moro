import pytest
import sympy as sp

from moro.abc import q1, q2
from moro.core import RigidBody2D, Robot


def assert_matrix_equal(a, b):
    diff = a - b
    assert all(sp.simplify(v) == 0 for v in diff)


def test_robot_initialization_and_basic_properties():
    robot = Robot((1, 0, 0, q1), (2, 0, 0, q2, "p"))

    assert robot.dof == 2
    assert robot.joint_types == ["r", "p"]
    assert robot.qi(1) == q1
    assert robot.qi(2) == 0
    assert str(robot) == "Robot RP"


def test_robot_forward_kinematics_and_frames():
    robot = Robot((1, 0, 0, q1),)

    expected_t = sp.Matrix(
        [
            [sp.cos(q1), -sp.sin(q1), 0, sp.cos(q1)],
            [sp.sin(q1), sp.cos(q1), 0, sp.sin(q1)],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    assert_matrix_equal(robot.T, expected_t) # Overall transformation from base to end-effector
    assert_matrix_equal(robot.T_i0(0), sp.eye(4)) # Identity since it's the base frame
    assert_matrix_equal(robot.T_ij(1, 1), sp.eye(4)) # Identity since it's the same frame
    assert_matrix_equal(robot.z(0), sp.Matrix([0, 0, 1])) # z-axis of the base frame
    assert_matrix_equal(robot.p(1), sp.Matrix([sp.cos(q1), sp.sin(q1), 0]))


def test_robot_geometric_jacobian_rr_planar():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))

    s12 = sp.sin(q1 + q2)
    c12 = sp.cos(q1 + q2)
    expected = sp.Matrix(
        [
            [-sp.sin(q1) - s12, -s12],
            [sp.cos(q1) + c12, c12],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ]
    )

    assert_matrix_equal(robot.J, expected)


def test_joint_limits_default_and_validation():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2, "p"))

    assert robot.joint_limits[0] == (-sp.pi, sp.pi)
    assert robot.joint_limits[1] == (0, 1000)

    Robot.joint_limits.fset(robot, (-1, 1), (0, 10))
    assert robot.joint_limits == ((-1, 1), (0, 10))

    with pytest.raises(ValueError, match="number of joint limits"):
        Robot.joint_limits.fset(robot, (-1, 1))

    with pytest.raises(ValueError, match="2-tuple"):
        Robot.joint_limits.fset(robot, (-1, 1), (0, 10, 20))


def test_robot_requires_dynamic_parameters_before_queries():
    robot = Robot((1, 0, 0, q1),)

    with pytest.raises(ValueError, match="Link masses are not defined"):
        robot.m_i(1)

    with pytest.raises(ValueError, match="Inertia tensors are not defined"):
        robot.I_i(1)

    with pytest.raises(ValueError, match="Center of mass locations are not defined"):
        robot.rcm_i(1)


def test_robot_center_of_mass_and_inertia_matrix_single_link():
    c, m, iz = sp.symbols("c m iz")
    robot = Robot((0, 0, 0, q1),)

    robot.set_cm_locations([(c, 0, 0)])
    robot.set_masses([m])
    robot.set_inertia_tensors([sp.diag(0, 0, iz)])

    expected_rcm = sp.Matrix([c * sp.cos(q1), c * sp.sin(q1), 0])
    expected_m = sp.Matrix([[c**2 * m + iz]])

    assert_matrix_equal(robot.rcm_i(1), expected_rcm)
    assert_matrix_equal(robot.get_inertia_matrix(), expected_m)


def test_rigid_body_2d_move_rotate_restart():
    rb = RigidBody2D([(1, 0), (0, 1), (0, 0)])

    rb.rotate(sp.pi / 2)
    p0 = rb.points[0]
    assert_matrix_equal(p0, sp.Matrix([0, 1, 0, 1]))

    rb.move([2, 0, 0])
    p0_translated = rb.points[0]
    assert_matrix_equal(p0_translated, sp.Matrix([0, 3, 0, 1]))

    rb.restart()
    assert_matrix_equal(rb.H, sp.eye(4))


def test_rigid_body_2d_centroid():
    rb = RigidBody2D([(0, 0), (2, 0), (0, 2)])

    cx, cy = rb.get_centroid()
    assert sp.simplify(cx - sp.Rational(2, 3)) == 0
    assert sp.simplify(cy - sp.Rational(2, 3)) == 0
