from importlib import import_module

import pytest
import sympy as sp
import warnings

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

    with pytest.raises(ValueError, match="Joint limit for joint 2 must be a 2-tuple"):
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


def test_warns_on_static_joint_variables_for_velocity_methods():
    q1s, q2s = sp.symbols("q1 q2")
    robot = Robot((1, 0, 0, q1s), (1, 0, 0, q2s))
    robot.masses = [1, 1]
    robot.inertia_tensors = None
    robot.cm_positions = [(0, 0, 0), (0, 0, 0)]
    with pytest.warns(UserWarning, match="time-dependent"):
        robot.w(1)
    with pytest.warns(UserWarning, match="time-dependent"):
        robot.v_cm(1)
    with pytest.warns(UserWarning, match="time-dependent"):
        robot.coriolis_matrix()


def test_no_warning_with_dynamicsymbols():
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    robot.masses = [1, 1]
    robot.inertia_tensors = None
    robot.cm_positions = [(0, 0, 0), (0, 0, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        robot.w(1)
        robot.v_cm(1)


def test_coriolis_and_dynamic_model_static_joint_variables_use_explicit_time_derivatives():
    q = sp.symbols("q")
    m, c, iz, g = sp.symbols("m c iz g")
    robot = Robot((0, 0, 0, q),)
    robot.masses = [m]
    robot.inertia_tensors = [sp.diag(0, 0, iz)]
    robot.cm_positions = [(c, 0, 0)]
    robot.gravity = (0, -g, 0)

    with pytest.warns(UserWarning, match="time-dependent"):
        C = robot.coriolis_matrix()
    assert C == sp.Matrix([[0]])

    with pytest.warns(UserWarning, match="dynamic_model"):
        equations = robot.dynamic_model()
    assert len(equations) == 1
    assert equations[0].rhs == sp.symbols("tau_1")


def test_m_validates_index():
    robot = Robot((0, 0, 0, q1), (0, 0, 0, q2))
    robot.masses = [1, 2]

    assert robot.m(1) == 1
    assert robot.m(robot.dof) == 2
    with pytest.raises(IndexError, match="link index 0 out of range"):
        robot.m(0)
    with pytest.raises(IndexError, match="link index 3 out of range"):
        robot.m(robot.dof + 1)
    with pytest.raises(TypeError, match="link index must be an integer"):
        robot.m(1.0)


def test_mutation_defensive_copies_for_masses_cm_positions_and_inertia_tensors():
    m1, m2, c, cc, iz = sp.symbols("m1 m2 c cc iz")
    robot = Robot((0, 0, 0, q1),)

    masses = [m1]
    robot.masses = masses
    masses[0] = m2
    assert robot.masses == [m1]
    exposed_masses = robot.masses
    exposed_masses[0] = m2
    assert robot.masses == [m1]

    positions = [sp.Matrix([[c, 0, 0]])]
    robot.cm_positions = positions
    positions[0][0, 0] = cc
    assert robot.cm_positions[0] == sp.Matrix([c, 0, 0])
    exposed_positions = robot.cm_positions
    exposed_positions[0][0, 0] = cc
    assert robot.cm_positions[0] == sp.Matrix([c, 0, 0])

    tensors = [sp.diag(0, 0, iz)]
    robot.inertia_tensors = tensors
    tensors[0][2, 2] = 99
    assert robot.inertia_tensors[0][2, 2] == iz
    exposed_tensors = robot.inertia_tensors
    exposed_tensors[0][2, 2] = 99
    assert robot.inertia_tensors[0][2, 2] == iz


def test_other_mutable_getters_do_not_expose_internal_containers():
    robot = Robot((1, 0, 0, q1),)

    dh_parameters = robot.dh_parameters
    dh_parameters[0] = (9, 9, 9, 9)
    assert robot.dh_parameters[0] == (1, 0, 0, q1)

    qs = robot.qs
    qs[0] = sp.symbols("other_q")
    assert robot.q(1) == q1

    limits = robot.joint_limits
    limits[0] = (-1, 1)
    assert robot.joint_limits[0] == (-sp.pi, sp.pi)


def test_joint_limits_valid_and_invalid_invariants():
    robot = Robot((1, 0, 0, q1), (1, 0, q2, 0, "p"))

    robot.joint_limits = [(-sp.pi / 2, sp.pi / 2), (0, 2.5)]
    assert robot.joint_limits == [(-sp.pi / 2, sp.pi / 2), (0, 2.5)]
    assert robot._numerical_joint_limits == [(-float(sp.pi / 2), float(sp.pi / 2)), (0.0, 2.5)]

    invalid_limits = [
        ([(-1, 1)], "number of joint limits"),
        ([(-1, 1), (0, 1, 2)], "joint 2"),
        ([(-1, 1), (sp.nan, 1)], "joint 2.*NaN"),
        ([(-1, 1), (-sp.oo, 1)], "joint 2.*infinite"),
        ([(-1, 1), (2, 1)], "joint 2.*lower <= upper"),
        ([(-1, 1), (sp.symbols("a"), 1)], "joint 2.*numeric"),
    ]
    for limits, message in invalid_limits:
        with pytest.raises(ValueError, match=message):
            robot.joint_limits = limits


def test_inertia_tensors_accept_convertible_3x3_and_reject_invalid_shapes():
    robot = Robot((0, 0, 0, q1),)

    matrix_tensor = sp.eye(3)
    robot.inertia_tensors = [matrix_tensor]
    assert robot.inertia_tensors[0] == sp.eye(3)

    nested_tensor = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    robot.inertia_tensors = [nested_tensor]
    assert robot.inertia_tensors[0] == sp.diag(1, 2, 3)

    for bad_tensor in (sp.Matrix([1, 2, 3]), [[1, 0], [0, 1]]):
        with pytest.raises(ValueError, match="link 1.*3x3"):
            robot.inertia_tensors = [bad_tensor]


def test_gravity_and_cm_positions_normalize_valid_vectors_and_reject_invalid_shapes():
    g, c = sp.symbols("g c")
    robot = Robot((0, 0, 0, q1),)

    for value in ((0, -g, 0), [0, -g, 0], sp.Matrix([0, -g, 0]), sp.Matrix([[0, -g, 0]])):
        robot.gravity = value
        assert robot.gravity == sp.Matrix([0, -g, 0])
        assert robot.gravity.shape == (3, 1)

    for value in ((c, 0, 0), [c, 0, 0], sp.Matrix([c, 0, 0]), sp.Matrix([[c, 0, 0]])):
        robot.cm_positions = [value]
        assert robot.cm_positions[0] == sp.Matrix([c, 0, 0])
        assert robot.cm_positions[0].shape == (3, 1)

    for bad_gravity in ((0, -g), sp.zeros(2, 2)):
        with pytest.raises(ValueError, match="Gravity acceleration.*three"):
            robot.gravity = bad_gravity
    for bad_cm in ((c, 0), sp.zeros(2, 2)):
        with pytest.raises(ValueError, match="Center of mass location for link 1.*three"):
            robot.cm_positions = [bad_cm]


def test_J_point_validates_and_normalizes_point_argument():
    robot = Robot((1, 0, 0, q1),)
    assert robot.J_point(sp.Matrix([[0, 0, 0]]), 1).shape == (6, 1)
    assert robot.J_point((0, 0, 0), 1).shape == (6, 1)

    for bad_point in ((0, 0), sp.zeros(2, 2)):
        with pytest.raises(ValueError, match="Point.*three"):
            robot.J_point(bad_point, 1)


def test_dh_row_structure_validation():
    robot4 = Robot((1, 0, 0, q1))
    assert robot4.dof == 1
    assert robot4.joint_type(1) == "r"

    robot5 = Robot((1, 0, q1, 0, "p"))
    assert robot5.dof == 1
    assert robot5.joint_type(1) == "p"

    with pytest.raises(ValueError, match="Invalid joint type"):
        Robot((1, 0, 0, q1, "x"))
    with pytest.raises(ValueError, match="row 1.*exactly 4 or 5"):
        Robot((1, 0, 0))
    with pytest.raises(ValueError, match="row 1.*exactly 4 or 5"):
        Robot((1, 0, 0, q1, "r", "extra"))
    with pytest.raises(ValueError, match="row 1 must be a list or tuple"):
        Robot("not-a-dh-row")
    with pytest.raises(ValueError, match="at least one DH"):
        Robot()
