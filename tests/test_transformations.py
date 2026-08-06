import sympy as sp
import pytest

from moro.transformations import axa2rot, eul2rot, rot2eul, rot2axa


def assert_matrix_equal(a, b):
    diff = a - b
    assert all(sp.simplify(v) == 0 for v in diff)


def assert_matrix_close(a, b, tol=1e-9):
    diff = sp.Matrix(a) - sp.Matrix(b)
    assert all(abs(float(sp.N(v))) <= tol for v in diff)


def assert_axis_angle_reconstructs(R, axis, angle):
    R_reconstructed = axa2rot(axis, angle)
    assert_matrix_equal(sp.simplify(R_reconstructed), R)


def test_rot2axa_identity_radians_and_degrees():
    R = sp.eye(3)

    axis, angle = rot2axa(R)
    assert_matrix_equal(axis, sp.Matrix([1, 0, 0]))
    assert sp.simplify(angle) == 0

    axis_deg, angle_deg = rot2axa(R, deg=True)
    assert_matrix_equal(axis_deg, sp.Matrix([1, 0, 0]))
    assert sp.simplify(angle_deg) == 0


def test_rot2axa_general_rotation_radians_and_degrees():
    original_axis = sp.Matrix([1, 2, 3])
    normalized_axis = sp.simplify(original_axis / original_axis.norm())
    original_angle = sp.pi / 3
    R = axa2rot(original_axis, original_angle)

    axis, angle = rot2axa(R)
    assert_matrix_equal(axis, normalized_axis)
    assert sp.simplify(angle - original_angle) == 0
    assert_axis_angle_reconstructs(R, axis, angle)

    axis_deg, angle_deg = rot2axa(R, deg=True)
    assert_matrix_equal(axis_deg, normalized_axis)
    assert sp.simplify(angle_deg - 60) == 0


def test_rot2axa_pi_rotation_cartesian_axes_reconstructs():
    for original_axis in (
        sp.Matrix([1, 0, 0]),
        sp.Matrix([0, 1, 0]),
        sp.Matrix([0, 0, 1]),
    ):
        R = axa2rot(original_axis, sp.pi)
        axis, angle = rot2axa(R)

        assert sp.simplify(axis.norm() - 1) == 0
        assert sp.simplify(angle - sp.pi) == 0
        assert_axis_angle_reconstructs(R, axis, angle)


def test_rot2axa_pi_rotation_negative_components_reconstructs():
    for original_axis in (
        sp.Matrix([1, -1, 0]),
        sp.Matrix([-1, 2, -3]),
    ):
        R = axa2rot(original_axis, sp.pi)
        axis, angle = rot2axa(R)

        assert sp.simplify(axis.norm() - 1) == 0
        assert sp.simplify(angle - sp.pi) == 0
        assert_axis_angle_reconstructs(R, axis, angle)


def test_rot2axa_pi_rotation_degrees():
    R = axa2rot(sp.Matrix([-1, 2, -3]), sp.pi)
    axis, angle = rot2axa(R, deg=True)

    assert sp.simplify(axis.norm() - 1) == 0
    assert sp.simplify(angle - 180) == 0


def test_rot2axa_pi_rotation_regression_preserves_relative_signs():
    original_axis = sp.Matrix([1, -1, 0])
    R = axa2rot(original_axis, sp.pi)

    axis, angle = rot2axa(R)


    assert sp.simplify(axis[0] * axis[1]) < 0
    assert_axis_angle_reconstructs(R, axis, angle)

    wrong_axis = sp.Matrix([1, 1, 0])
    wrong_reconstruction = axa2rot(wrong_axis, sp.pi)
    assert any(sp.simplify(v) != 0 for v in wrong_reconstruction - R)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_positive_singularity_exact(seq):
    R = eul2rot(sp.pi / 5, 0, sp.pi / 7, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert sp.simplify(theta) == 0
    assert sp.simplify(psi) == 0
    assert_matrix_equal(eul2rot(phi, theta, psi, seq=seq), R)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_negative_singularity_exact(seq):
    R = eul2rot(sp.pi / 5, sp.pi, sp.pi / 7, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert sp.simplify(theta - sp.pi) == 0
    assert sp.simplify(psi) == 0
    assert_matrix_equal(eul2rot(phi, theta, psi, seq=seq), R)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_near_positive_singularity_float(seq):
    R = sp.Matrix(sp.N(eul2rot(0.3, 0, 0.4, seq=seq)))
    R[2, 2] = sp.Float("0.9999999999999998")

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert theta == 0
    assert psi == 0
    R_clipped = R.copy()
    R_clipped[2, 2] = 1.0
    assert_matrix_close(eul2rot(phi, theta, psi, seq=seq), R_clipped, tol=1e-9)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_near_negative_singularity_float(seq):
    R = sp.Matrix(sp.N(eul2rot(0.3, sp.pi, 0.4, seq=seq)))
    R[2, 2] = sp.Float("-0.9999999999999998")

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert theta == sp.pi
    assert psi == 0
    R_clipped = R.copy()
    R_clipped[2, 2] = -1.0
    assert_matrix_close(eul2rot(phi, theta, psi, seq=seq), R_clipped, tol=1e-9)


@pytest.mark.parametrize("seq,value,expected", [
    ("zxz", sp.Float("1.000000000001"), 1.0),
    ("zyz", sp.Float("1.000000000001"), 1.0),
    ("zxz", sp.Float("-1.000000000001"), -1.0),
    ("zyz", sp.Float("-1.000000000001"), -1.0),
])
def test_rot2eul_clips_slightly_out_of_range_r33(seq, value, expected):
    R = sp.Matrix(sp.N(eul2rot(0.3, 0 if expected > 0 else sp.pi, 0.4, seq=seq)))
    R[2, 2] = value

    solutions = rot2eul(R, seq=seq, tol=1e-9)

    assert len(solutions) == 1
    assert not any(sp.sympify(angle).has(sp.I) for solution in solutions for angle in solution)


@pytest.mark.parametrize("seq,value", [
    ("zxz", sp.Float("1.0001")),
    ("zyz", sp.Float("1.0001")),
    ("zxz", sp.Float("-1.0001")),
    ("zyz", sp.Float("-1.0001")),
])
def test_rot2eul_rejects_r33_outside_tolerance(seq, value):
    R = sp.eye(3)
    R[2, 2] = value

    with pytest.raises(ValueError, match="outside the valid range"):
        rot2eul(R, seq=seq, tol=1e-9)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_general_numeric_returns_two_real_reconstructing_solutions(seq):
    R = sp.N(eul2rot(0.3, 0.8, -0.4, seq=seq))

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 2
    for solution in solutions:
        assert not any(sp.sympify(angle).has(sp.I) for angle in solution)
        assert_matrix_close(eul2rot(*solution, seq=seq), R, tol=1e-9)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_exact_symbolic_values_do_not_introduce_floats(seq):
    R = eul2rot(sp.pi / 3, sp.pi / 4, sp.pi / 6, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 2
    assert not any(angle.has(sp.Float) for solution in solutions for angle in solution)


@pytest.mark.parametrize("seq", ["zxz", "zyz"])
def test_rot2eul_completely_symbolic_matrix_does_not_raise_boolean_error(seq):
    phi, theta, psi = sp.symbols("phi theta psi", real=True)
    R = eul2rot(phi, theta, psi, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) in (1, 2)
    assert not any(angle.has(sp.Float) for solution in solutions for angle in solution)


@pytest.mark.parametrize("tol", [0, -1e-9])
def test_rot2eul_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol must be greater than 0"):
        rot2eul(sp.eye(3), tol=tol)


@pytest.mark.parametrize("R", [sp.eye(2), sp.zeros(3, 4), [[1, 0], [0, 1]]])
def test_rot2eul_invalid_shape(R):
    with pytest.raises(ValueError, match="3x3"):
        rot2eul(R)