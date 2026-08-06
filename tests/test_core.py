from importlib import import_module

import pytest
import sympy as sp

from moro.abc import q1, q2
from moro.core import Robot

RigidBody2D = None
try:
    RigidBody2D = import_module("examples.ejemplo01").RigidBody2D
except (ImportError, ModuleNotFoundError):
    pass


def assert_matrix_equal(a, b):
    diff = a - b
    assert all(sp.simplify(v) == 0 for v in diff)


def test_robot_initialization_and_basic_properties():
    robot = Robot((1, 0, 0, q1), (2, 0, 0, q2, "p"))

    assert robot.dof == 2
    assert robot.joint_types == ["r", "p"]
    assert robot.q(1) == q1
    assert robot.q(2) == 0
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
    assert_matrix_equal(robot.r_o(1), sp.Matrix([sp.cos(q1), sp.sin(q1), 0]))


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

    Robot.joint_limits.fset(robot, [(-1, 1), (0, 10)])
    assert robot.joint_limits == [(-1, 1), (0, 10)]

    with pytest.raises(ValueError, match="The number of joint limits must match DOF"):
        Robot.joint_limits.fset(robot, [(-1, 1)])

    with pytest.raises(ValueError, match="Each joint-limit should be a 2-tuple"):
        Robot.joint_limits.fset(robot, [(-1, 1), (0, 10, 20)])

def test_robot_center_of_mass_and_inertia_matrix_single_link():
    c, m, iz = sp.symbols("c m iz")
    robot = Robot((0, 0, 0, q1),)

    robot.cm_positions = [(c, 0, 0)]
    robot.masses = [m]
    robot.inertia_tensors = [sp.diag(0, 0, iz)]

    expected_rcm = sp.Matrix([c * sp.cos(q1), c * sp.sin(q1), 0])
    expected_m = sp.Matrix([[c**2 * m + iz]])

    assert_matrix_equal(robot.r_cm(1), expected_rcm)
    assert_matrix_equal(robot.inertia_matrix(), expected_m)





def test_r_cm_cache_reflects_cm_positions_changes():
    c, cc = sp.symbols("c cc")
    robot = Robot((0, 0, 0, q1),)
    robot.masses = [1]
    robot.cm_positions = [(c, 0, 0)]
    v1 = robot.r_cm(1)
    # Change the center of mass location
    robot.cm_positions = [(cc, 0, 0)]
    v2 = robot.r_cm(1)
    # The cached r_cm must reflect the new CoM location
    assert v2[0].has(cc)
    assert not v2[0].has(c)
    # J_cm family must also reflect the change
    assert robot.J_cm_i(1)[0, 0].has(cc)
    assert not robot.J_cm_i(1)[0, 0].has(c)


def test_inertia_tensors_none_auto_generates_diagonal():
    robot = Robot((0, 0, 0, q1),)
    robot.masses = [sp.symbols("m")]
    robot.cm_positions = [(sp.symbols("c"), 0, 0)]
    # Passing None must auto-generate diagonal tensors instead of crashing
    robot.inertia_tensors = None
    assert robot.inertia_tensors is not None
    assert len(robot.inertia_tensors) == robot.dof
    assert robot.inertia_tensors[0].shape == (3, 3)
    # The private generator persists the state (internal helper, no return)
    robot._inertia_tensors = None
    assert robot._generate_diagonal_inertia_tensors() is None
    assert robot._inertia_tensors is not None
    assert len(robot._inertia_tensors) == robot.dof
    assert robot.inertia_matrix().shape == (1, 1)


def test_joint_type_validation_and_case_insensitivity():
    # Uppercase joint types are accepted and normalized to lowercase
    robot = Robot((1, 0, 0, q1, "R"), (1, 0, 0, q2, "P"))
    assert robot.joint_type(1) == "r"
    assert robot.joint_type(2) == "p"
    assert robot.joint_limits[0] == (-sp.pi, sp.pi)
    assert robot.joint_limits[1] == (0, 1000)
    # Invalid joint types must raise a clear error
    with pytest.raises(ValueError, match="Invalid joint type"):
        Robot((1, 0, 0, q1, "giratorio"))
    with pytest.raises(ValueError, match="Invalid joint type"):
        Robot((1, 0, 0, q1, "x"))


def test_model_summary_reports_explicit_vs_assumed():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    s = robot.model_summary()
    # Nothing dynamic is defined yet
    assert "masses" in s and "NOT SET" in s
    assert "inertia_tensors" in s and "NOT SET" in s
    assert "cm_positions" in s and "NOT SET" in s
    assert "gravity" in s and "NOT SET" in s
    assert "joint_limits     : default" in s
    # Explicit masses + auto (assumed) diagonal inertia
    robot.masses = [1, 2]
    robot.inertia_tensors = None
    s = robot.model_summary()
    assert "masses           : explicit" in s
    assert "inertia_tensors  : assumed (diagonal symbolic)" in s
    # Custom limits are reported
    robot.joint_limits = [(-1, 1), (0, 5)]
    s = robot.model_summary()
    assert "joint_limits     : custom" in s
    # Explicit CoM and gravity
    robot.cm_positions = [(0, 0, 0), (0, 0, 0)]
    robot.gravity = (0, -9.81, 0)
    s = robot.model_summary()
    assert "cm_positions     : explicit" in s
    assert "gravity          : explicit" in s


def test_qis_range_requires_setting():
    robot = Robot((1, 0, 0, q1),)
    # Reading before setting must raise a clear error instead of AttributeError
    with pytest.raises(ValueError, match="qis_range has not been set"):
        robot.qis_range
    robot.qis_range = ((-1, 1),)
    assert robot.qis_range == ((-1, 1),)


def test_cm_positions_accepts_tuples():
    c = sp.symbols("c")
    robot = Robot((0, 0, 0, q1),)
    # Immutable tuples must be accepted (previously raised item-assignment TypeError)
    robot.cm_positions = ((c, 0, 0),)
    assert robot.cm_positions[0] == sp.Matrix([c, 0, 0])
    assert robot.r_cm(1)[0].has(c)


def test_T_ij_analytic_inverse_composes_to_identity():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    T_2_0 = robot.T_ij(2, 0)  # forward composition
    T_0_2 = robot.T_ij(0, 2)  # analytic (fast) inverse
    # The analytic inverse must be a true inverse: T_0_2 * T_2_0 = I
    assert_matrix_equal(T_0_2 * T_2_0, sp.eye(4))
