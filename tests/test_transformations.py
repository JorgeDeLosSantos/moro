import sympy as sp
import pytest

from moro.transformations import _rot2htm, axa2rot, eul2rot, htmrot, htmtra, rot, rot2eul, rot2axa


PROPER_EULER_SEQUENCES = ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]
EULER_COS_INDEX = {
    "xyx": (0, 0),
    "xzx": (0, 0),
    "yxy": (1, 1),
    "yzy": (1, 1),
    "zxz": (2, 2),
    "zyz": (2, 2),
}


def assert_matrix_equal(a, b):
    diff = a - b
    assert all(sp.simplify(v) == 0 for v in diff)


def assert_matrix_close(a, b, tol=1e-9):
    diff = sp.Matrix(a) - sp.Matrix(b)
    assert all(abs(float(sp.N(v))) <= tol for v in diff)


def assert_axis_angle_reconstructs(R, axis, angle):
    R_reconstructed = axa2rot(axis, angle)
    assert_matrix_equal(sp.simplify(R_reconstructed), R)


def assert_axis_angle_reconstructs_close(R, axis, angle, deg=False, tol=1e-8):
    if deg:
        angle = angle * sp.pi / 180
    R_reconstructed = axa2rot(axis, angle)
    assert_matrix_close(R_reconstructed, R, tol=tol)


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


@pytest.mark.parametrize("axis", [
    sp.Matrix([1, 0, 0]),
    sp.Matrix([1, -1, 0]),
    sp.Matrix([-1, 2, -3]),
])
def test_rot2axa_exact_symbolic_pi_cases_reconstruct(axis):
    R = axa2rot(axis, sp.pi)

    recovered_axis, angle = rot2axa(R)

    assert sp.simplify(angle - sp.pi) == 0
    assert sp.simplify(recovered_axis.norm() - 1) == 0
    assert_axis_angle_reconstructs(R, recovered_axis, angle)


@pytest.mark.parametrize("angle", [float(sp.pi), float(sp.pi) - 1e-10])
def test_rot2axa_numeric_near_pi_uses_stable_branch(angle):
    original_axis = sp.Matrix([1, -1, 0])
    R = axa2rot(original_axis, angle)

    axis, recovered_angle = rot2axa(R)

    assert abs(float(sp.N(recovered_angle)) - float(sp.pi)) <= 1e-9
    assert float(sp.N(axis[0] * axis[1])) < 0
    assert_axis_angle_reconstructs_close(R, axis, recovered_angle, tol=1e-8)


def test_rot2axa_numeric_near_zero_uses_identity_branch():
    R = axa2rot(sp.Matrix([1, 2, 3]), 1e-10)

    axis, angle = rot2axa(R)

    assert_matrix_equal(axis, sp.Matrix([1, 0, 0]))
    assert angle == 0
    assert_axis_angle_reconstructs_close(R, axis, angle, tol=1e-8)


def test_rot2axa_numeric_general_float_reconstructs():
    R = axa2rot(sp.Matrix([1, 2, 3]), 0.7)

    axis, angle = rot2axa(R)

    assert_axis_angle_reconstructs_close(R, axis, angle, tol=1e-9)


def test_rot2axa_numeric_degrees_reconstructs():
    R = axa2rot(sp.Matrix([1, 2, 3]), 0.7)

    axis, angle_deg = rot2axa(R, deg=True)

    assert abs(float(sp.N(angle_deg)) - float(0.7 * 180 / sp.pi)) <= 1e-9
    assert_axis_angle_reconstructs_close(R, axis, angle_deg, deg=True, tol=1e-9)


@pytest.mark.parametrize("tol", [0, -1e-9])
def test_rot2axa_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol must be greater than 0"):
        rot2axa(sp.eye(3), tol=tol)


def test_rot2axa_numeric_pi_regression_does_not_use_unstable_general_branch():
    R = axa2rot(sp.Matrix([1, -1, 0]), float(sp.pi))

    axis, angle = rot2axa(R)

    assert abs(float(sp.N(angle)) - float(sp.pi)) <= 1e-9
    assert float(sp.N(axis[0] * axis[1])) < 0
    assert_axis_angle_reconstructs_close(R, axis, angle, tol=1e-8)


@pytest.mark.parametrize("args,expected", [
    ((), sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
    ((1,), sp.Matrix([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
    ((1, 2), sp.Matrix([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]])),
    ((1, 2, 3), sp.Matrix([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])),
])
def test_htmtra_positional_api(args, expected):
    assert_matrix_equal(htmtra(*args), expected)


def test_htmtra_keyword_api():
    assert_matrix_equal(htmtra(x=1, y=2, z=3), sp.Matrix([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]))
    assert_matrix_equal(htmtra(z=5), sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]]))


def test_htmtra_symbolic_values():
    a, b, c = sp.symbols("a b c")
    assert_matrix_equal(htmtra(x=a, y=b, z=c), sp.Matrix([[1, 0, 0, a], [0, 1, 0, b], [0, 0, 1, c], [0, 0, 0, 1]]))


@pytest.mark.parametrize("call", [
    lambda: htmtra(1, 2, 3, 4),
    lambda: htmtra(dx=1),
    lambda: htmtra(foo=1),
    lambda: htmtra([1, 2, 3]),
])
def test_htmtra_rejects_old_or_invalid_api(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("lower,upper", [("x", "X"), ("y", "Y"), ("z", "Z")])
def test_rot_and_htmrot_accept_case_insensitive_text_axes(lower, upper):
    theta = sp.pi / 4
    assert_matrix_equal(rot(theta, lower), rot(theta, upper))
    assert_matrix_equal(htmrot(theta, lower), htmrot(theta, upper))


@pytest.mark.parametrize("bad_axis", [1, "1", "xy", "", None])
def test_rot_and_htmrot_reject_invalid_axes_with_value_error(bad_axis):
    with pytest.raises(ValueError, match="axis must be 'x', 'y' or 'z'"):
        rot(sp.pi / 4, bad_axis)
    with pytest.raises(ValueError, match="axis must be 'x', 'y' or 'z'"):
        htmrot(sp.pi / 4, bad_axis)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("theta,deg", [(sp.pi / 6, False), (30, True)])
def test_htmrot_reuses_rot_equivalence(axis, theta, deg):
    assert_matrix_equal(htmrot(theta, axis=axis, deg=deg), _rot2htm(rot(theta, axis=axis, deg=deg)))


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_positive_singularity_exact(seq):
    R = eul2rot(sp.pi / 5, 0, sp.pi / 7, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert sp.simplify(theta) == 0
    assert sp.simplify(psi) == 0
    assert_matrix_equal(eul2rot(phi, theta, psi, seq=seq), R)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_negative_singularity_exact(seq):
    R = eul2rot(sp.pi / 5, sp.pi, sp.pi / 7, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert sp.simplify(theta - sp.pi) == 0
    assert sp.simplify(psi) == 0
    assert_matrix_equal(eul2rot(phi, theta, psi, seq=seq), R)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_near_positive_singularity_float(seq):
    R = sp.Matrix(sp.N(eul2rot(0.3, 0, 0.4, seq=seq)))
    i, j = EULER_COS_INDEX[seq]
    R[i, j] = sp.Float("0.9999999999999998")

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert theta == 0
    assert psi == 0
    R_clipped = R.copy()
    R_clipped[i, j] = 1.0
    assert_matrix_close(eul2rot(phi, theta, psi, seq=seq), R_clipped, tol=1e-9)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_near_negative_singularity_float(seq):
    R = sp.Matrix(sp.N(eul2rot(0.3, sp.pi, 0.4, seq=seq)))
    i, j = EULER_COS_INDEX[seq]
    R[i, j] = sp.Float("-0.9999999999999998")

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, theta, psi = solutions[0]
    assert theta == sp.pi
    assert psi == 0
    R_clipped = R.copy()
    R_clipped[i, j] = -1.0
    assert_matrix_close(eul2rot(phi, theta, psi, seq=seq), R_clipped, tol=1e-9)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
@pytest.mark.parametrize("value,expected", [(sp.Float("1.000000000001"), 1.0), (sp.Float("-1.000000000001"), -1.0)])
def test_rot2eul_clips_slightly_out_of_range_r33(seq, value, expected):
    R = sp.Matrix(sp.N(eul2rot(0.3, 0 if expected > 0 else sp.pi, 0.4, seq=seq)))
    i, j = EULER_COS_INDEX[seq]
    R[i, j] = value

    solutions = rot2eul(R, seq=seq, tol=1e-9)

    assert len(solutions) == 1
    assert not any(sp.sympify(angle).has(sp.I) for solution in solutions for angle in solution)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
@pytest.mark.parametrize("value", [sp.Float("1.0001"), sp.Float("-1.0001")])
def test_rot2eul_rejects_r33_outside_tolerance(seq, value):
    R = sp.eye(3)
    i, j = EULER_COS_INDEX[seq]
    R[i, j] = value

    with pytest.raises(ValueError, match="outside the valid range"):
        rot2eul(R, seq=seq, tol=1e-9)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_general_numeric_returns_two_real_reconstructing_solutions(seq):
    R = sp.N(eul2rot(0.3, 0.8, -0.4, seq=seq))

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 2
    for solution in solutions:
        assert not any(sp.sympify(angle).has(sp.I) for angle in solution)
        assert_matrix_close(eul2rot(*solution, seq=seq), R, tol=1e-9)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_exact_symbolic_values_do_not_introduce_floats(seq):
    R = eul2rot(sp.pi / 3, sp.pi / 4, sp.pi / 6, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 2
    assert not any(angle.has(sp.Float) for solution in solutions for angle in solution)
    for solution in solutions:
        assert_matrix_equal(eul2rot(*solution, seq=seq), R)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
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


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_general_degrees_reconstructs(seq):
    R = eul2rot(30, 45, 60, seq=seq, deg=True)

    solutions = rot2eul(R, seq=seq, deg=True)

    assert len(solutions) == 2
    for solution in solutions:
        assert_matrix_close(eul2rot(*solution, seq=seq, deg=True), R, tol=1e-9)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
@pytest.mark.parametrize("theta", [1e-10, float(sp.pi) - 1e-10])
def test_rot2eul_near_singularities_classified_with_default_tolerance(seq, theta):
    R = sp.N(eul2rot(0.3, theta, 0.4, seq=seq))

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    assert solutions[0][2] == 0
    assert_matrix_close(eul2rot(*solutions[0], seq=seq), R, tol=1e-8)


@pytest.mark.parametrize("seq", PROPER_EULER_SEQUENCES)
def test_rot2eul_near_singularity_can_be_general_with_smaller_tolerance(seq):
    R = sp.N(eul2rot(0.3, 1e-6, 0.4, seq=seq))

    solutions = rot2eul(R, seq=seq, tol=1e-14)

    assert len(solutions) == 2
    for solution in solutions:
        assert_matrix_close(eul2rot(*solution, seq=seq), R, tol=1e-8)


@pytest.mark.parametrize("seq_lower,seq_mixed,seq_upper", [("zxz", "ZxZ", "ZXZ"), ("xyx", "XyX", "XYX")])
def test_euler_sequence_case_insensitive(seq_lower, seq_mixed, seq_upper):
    R_lower = eul2rot(sp.pi / 6, sp.pi / 4, sp.pi / 3, seq=seq_lower)
    R_mixed = eul2rot(sp.pi / 6, sp.pi / 4, sp.pi / 3, seq=seq_mixed)
    R_upper = eul2rot(sp.pi / 6, sp.pi / 4, sp.pi / 3, seq=seq_upper)

    assert_matrix_equal(R_lower, R_mixed)
    assert_matrix_equal(R_lower, R_upper)
    assert rot2eul(R_lower, seq=seq_lower) == rot2eul(R_lower, seq=seq_mixed)
    assert rot2eul(R_lower, seq=seq_lower) == rot2eul(R_lower, seq=seq_upper)


@pytest.mark.parametrize("seq", ["xyz", "zyx", "", 1, None])
def test_euler_invalid_sequences_raise_value_error(seq):
    with pytest.raises(ValueError, match="seq must be one of"):
        eul2rot(0, 0, 0, seq=seq)
    with pytest.raises(ValueError, match="seq must be one of"):
        rot2eul(sp.eye(3), seq=seq)