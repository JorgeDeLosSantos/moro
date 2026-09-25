import sympy as sp
import pytest

from moro.transformations import (
    axa2quat,
    axa2rot,
    eul2rot,
    htm2rot,
    htm2tra,
    htmrot,
    htmtra,
    invhtm,
    is_rotation_matrix,
    quat2axa,
    quat2rot,
    rot,
    rot2eul,
    rot2quat,
    rot2rotvec,
    rot2axa,
    rot2htm,
    rt2htm,
    skew,
)


PROPER_EULER_SEQUENCES = ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]
TAIT_BRYAN_SEQUENCES = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
ALL_EULER_SEQUENCES = PROPER_EULER_SEQUENCES + TAIT_BRYAN_SEQUENCES
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


@pytest.mark.parametrize("axis", [
    [0, 0, 1],
    (0, 0, 1),
    sp.Matrix([0, 0, 1]),
    sp.Matrix([[0, 0, 1]]),
])
def test_axa2rot_accepts_supported_vector_formats(axis):
    assert_matrix_equal(axa2rot(axis, sp.pi / 3), rot(sp.pi / 3, "z"))


def test_axa2rot_accepts_non_normalized_axis():
    assert_matrix_equal(axa2rot([0, 0, 5], sp.pi / 3), rot(sp.pi / 3, "z"))


def test_axa2rot_symbolic_axis_does_not_reject_undecidable_zero_norm():
    kx, ky, kz, theta = sp.symbols("kx ky kz theta")
    R = axa2rot(sp.Matrix([kx, ky, kz]), theta)

    assert R.shape == (3, 3)
    assert R.has(kx, ky, kz, theta)


def test_axa2rot_rejects_zero_axis():
    with pytest.raises(ValueError, match="The rotation axis cannot be the zero vector"):
        axa2rot([0, 0, 0], sp.pi / 3)


@pytest.mark.parametrize("axis", [
    [],
    [1, 2],
    [1, 2, 3, 4],
    sp.Matrix([[1, 2], [3, 4]]),
    sp.Matrix([[1], [2]]),
    sp.Matrix([[1], [2], [3], [4]]),
])
def test_axa2rot_rejects_invalid_axis_dimensions(axis):
    with pytest.raises(ValueError, match="3D vector"):
        axa2rot(axis, sp.pi / 3)


@pytest.mark.parametrize("axis,theta", [
    ([1, 0, 0], sp.pi / 3),
    ([1, 2, 3], sp.pi),
    ([1, -2, 3], 0.7),
])
def test_axa2rot_rot2axa_reconstruction(axis, theta):
    R = axa2rot(axis, theta)
    recovered_axis, angle = rot2axa(R)
    R2 = axa2rot(recovered_axis, angle)

    if any(value.has(sp.Float) for value in sp.Matrix(R)):
        assert_matrix_close(R2, R, tol=1e-9)
    else:
        assert_matrix_equal(R2, R)


@pytest.mark.parametrize("u", [
    [sp.Symbol("ux"), sp.Symbol("uy"), sp.Symbol("uz")],
    (sp.Symbol("ux"), sp.Symbol("uy"), sp.Symbol("uz")),
    sp.Matrix([sp.Symbol("ux"), sp.Symbol("uy"), sp.Symbol("uz")]),
    sp.Matrix([[sp.Symbol("ux"), sp.Symbol("uy"), sp.Symbol("uz")]]),
])
def test_skew_accepts_supported_vector_formats(u):
    ux, uy, uz = sp.symbols("ux uy uz")
    expected = sp.Matrix([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0],
    ])

    S = skew(u)

    assert_matrix_equal(S, expected)
    assert_matrix_equal(S.T, -S)


@pytest.mark.parametrize("u", [
    [],
    [1, 2],
    [1, 2, 3, 4],
    sp.Matrix([[1, 2], [3, 4]]),
    sp.Matrix([[1], [2]]),
    sp.Matrix([[1], [2], [3], [4]]),
])
def test_skew_rejects_invalid_vector_dimensions(u):
    with pytest.raises(ValueError, match="3D vector"):
        skew(u)


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
    assert_matrix_equal(htmrot(theta, axis=axis, deg=deg), rot2htm(rot(theta, axis=axis, deg=deg)))


def test_rot2htm_builds_expected_homogeneous_matrix():
    R = rot(sp.pi / 3, "z")
    T = rot2htm(R)

    assert T.shape == (4, 4)
    assert_matrix_equal(T[:3, :3], R)
    assert_matrix_equal(T[:3, 3], sp.zeros(3, 1))
    assert_matrix_equal(T[3, :], sp.Matrix([[0, 0, 0, 1]]))


@pytest.mark.parametrize("R", [sp.eye(2), sp.zeros(3, 4), [[1, 0], [0, 1]]])
def test_rot2htm_rejects_invalid_shape(R):
    with pytest.raises(ValueError, match="3x3"):
        rot2htm(R)


@pytest.mark.parametrize("p", [
    [1, 2, 3],
    (1, 2, 3),
    sp.Matrix([1, 2, 3]),
    sp.Matrix([[1, 2, 3]]),
])
def test_rt2htm_numeric_vector_formats_round_trip(p):
    R = rot(sp.pi / 3, "z")
    p_column = sp.Matrix([1, 2, 3])
    T = rt2htm(R, p)

    assert T.shape == (4, 4)
    assert_matrix_equal(htm2rot(T), R)
    assert_matrix_equal(htm2tra(T), p_column)


def test_rt2htm_symbolic_round_trip():
    theta, px, py, pz = sp.symbols("theta px py pz")
    R = rot(theta, "x")
    p = sp.Matrix([px, py, pz])
    T = rt2htm(R, p)

    assert_matrix_equal(htm2rot(T), R)
    assert_matrix_equal(htm2tra(T), p)


def test_rt2htm_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="3x3"):
        rt2htm(sp.eye(2), [1, 2, 3])
    with pytest.raises(ValueError, match="3D vector"):
        rt2htm(sp.eye(3), [1, 2])


def test_htm2rot_and_htm2tra_extract_blocks():
    R = rot(sp.pi / 4, "y")
    p = sp.Matrix([4, 5, 6])
    T = rt2htm(R, p)

    assert_matrix_equal(htm2rot(T), R)
    assert htm2tra(T).shape == (3, 1)
    assert_matrix_equal(htm2tra(T), p)


@pytest.mark.parametrize("T", [sp.eye(3), sp.zeros(4, 3), [[1, 0], [0, 1]]])
def test_htm2rot_and_htm2tra_reject_invalid_shape(T):
    with pytest.raises(ValueError, match="4x4"):
        htm2rot(T)
    with pytest.raises(ValueError, match="4x4"):
        htm2tra(T)


@pytest.mark.parametrize("R,p", [
    (rot(0.7, "z") * rot(0.3, "x"), sp.Matrix([1.2, -3.4, 5.6])),
    (rot(sp.Symbol("theta"), "z"), sp.Matrix(sp.symbols("px py pz"))),
])
def test_invhtm_structured_inverse_matches_identity_and_matrix_inverse(R, p):
    T = rt2htm(R, p)
    T_inv = invhtm(T)

    assert_matrix_close(sp.simplify(T * T_inv), sp.eye(4), tol=1e-9)
    assert_matrix_close(sp.simplify(T_inv * T), sp.eye(4), tol=1e-9)
    if any(value.has(sp.Float) for value in sp.Matrix(T)):
        assert_matrix_close(T_inv, T.inv(), tol=1e-9)
    else:
        assert_matrix_equal(sp.simplify(T_inv), sp.simplify(T.inv()))


def test_invhtm_rejects_invalid_shape():
    with pytest.raises(ValueError, match="4x4"):
        invhtm(sp.eye(3))


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
@pytest.mark.parametrize("theta", [1e-10, float(sp.pi) - 1e-10])
def test_rot2eul_valid_near_proper_euler_singularity(seq, theta):
    R = sp.N(eul2rot(0.3, theta, 0.4, seq=seq))

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    assert solutions[0][2] == 0
    assert_matrix_close(eul2rot(*solutions[0], seq=seq), R, tol=1e-8)


def test_rot2eul_rejects_non_rotation_matrix():
    R = sp.eye(3)
    R[0, 0] = sp.Float("1.0000001")

    with pytest.raises(ValueError, match="rotation matrix"):
        rot2eul(R)


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


@pytest.mark.parametrize("seq", ["xxx", "xy", "", "abc"])
def test_euler_invalid_sequence_strings_raise_value_error(seq):
    with pytest.raises(ValueError, match="seq must be one of"):
        eul2rot(0, 0, 0, seq=seq)
    with pytest.raises(ValueError, match="seq must be one of"):
        rot2eul(sp.eye(3), seq=seq)


@pytest.mark.parametrize("seq", [1, None, [], object()])
def test_euler_non_string_sequences_raise_type_error(seq):
    with pytest.raises(TypeError, match="seq must be a string"):
        eul2rot(0, 0, 0, seq=seq)
    with pytest.raises(TypeError, match="seq must be a string"):
        rot2eul(sp.eye(3), seq=seq)


@pytest.mark.parametrize("seq", TAIT_BRYAN_SEQUENCES)
def test_tait_bryan_general_numeric_returns_two_reconstructing_solutions(seq):
    R = sp.N(eul2rot(0.3, 0.5, -0.4, seq=seq))

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 2
    for solution in solutions:
        assert_matrix_close(eul2rot(*solution, seq=seq), R, tol=1e-9)


@pytest.mark.parametrize("seq", TAIT_BRYAN_SEQUENCES)
@pytest.mark.parametrize("theta", [sp.pi / 2, -sp.pi / 2])
def test_tait_bryan_exact_singularities_reconstruct_with_third_angle_zero(seq, theta):
    R = eul2rot(sp.pi / 5, theta, sp.pi / 7, seq=seq)

    solutions = rot2eul(R, seq=seq)

    assert len(solutions) == 1
    phi, recovered_theta, psi = solutions[0]
    assert sp.simplify(recovered_theta - theta) == 0
    assert sp.simplify(psi) == 0
    assert_matrix_equal(eul2rot(phi, recovered_theta, psi, seq=seq), R)


@pytest.mark.parametrize("seq", ALL_EULER_SEQUENCES)
def test_euler_extrinsic_general_round_trip(seq):
    R = sp.N(eul2rot(0.2, 0.4, -0.3, seq=seq, intrinsic=False))

    solutions = rot2eul(R, seq=seq, intrinsic=False)

    assert len(solutions) == 2
    for solution in solutions:
        assert_matrix_close(
            eul2rot(*solution, seq=seq, intrinsic=False),
            R,
            tol=1e-9,
        )


@pytest.mark.parametrize("seq", TAIT_BRYAN_SEQUENCES)
@pytest.mark.parametrize("theta", [sp.pi / 2, -sp.pi / 2])
def test_tait_bryan_extrinsic_singular_round_trip_keeps_public_third_angle_zero(seq, theta):
    R = eul2rot(sp.pi / 5, theta, sp.pi / 7, seq=seq, intrinsic=False)

    solutions = rot2eul(R, seq=seq, intrinsic=False)

    assert len(solutions) == 1
    assert sp.simplify(solutions[0][2]) == 0
    assert_matrix_equal(
        eul2rot(*solutions[0], seq=seq, intrinsic=False),
        R,
    )


@pytest.mark.parametrize("seq", ALL_EULER_SEQUENCES)
def test_euler_intrinsic_extrinsic_equivalence(seq):
    phi, theta, psi = sp.pi / 7, sp.pi / 5, -sp.pi / 9

    R_ext = eul2rot(phi, theta, psi, seq=seq, intrinsic=False)
    R_int = eul2rot(psi, theta, phi, seq=seq[::-1], intrinsic=True)

    assert_matrix_equal(R_ext, R_int)


@pytest.mark.parametrize("intrinsic", [0, 1, None, "yes"])
def test_euler_intrinsic_requires_bool(intrinsic):
    with pytest.raises(TypeError, match="intrinsic must be a bool"):
        eul2rot(0, 0, 0, intrinsic=intrinsic)
    with pytest.raises(TypeError, match="intrinsic must be a bool"):
        rot2eul(sp.eye(3), intrinsic=intrinsic)


def test_is_rotation_matrix_numeric_and_symbolic_contract():
    theta = sp.symbols("theta", real=True)

    assert is_rotation_matrix(sp.eye(3)) is True
    assert is_rotation_matrix(rot(theta, "z")) is True
    assert is_rotation_matrix(sp.diag(1, 1, -1)) is False
    assert is_rotation_matrix(sp.eye(2)) is False

    a, b, c, d, e, f, g, h, i = sp.symbols("a:i")
    generic = sp.Matrix([[a, b, c], [d, e, f], [g, h, i]])
    assert is_rotation_matrix(generic) is None


def test_rot2eul_rejects_symbolically_indeterminate_matrix():
    a, b, c, d, e, f, g, h, i = sp.symbols("a:i")
    generic = sp.Matrix([[a, b, c], [d, e, f], [g, h, i]])

    with pytest.raises(ValueError, match="indeterminate"):
        rot2eul(generic)



def assert_quaternion_equivalent(q1, q2, tol=1e-9):
    q1 = sp.Matrix(q1)
    q2 = sp.Matrix(q2)
    try:
        assert_matrix_close(q1, q2, tol=tol)
        return
    except AssertionError:
        pass
    assert_matrix_close(q1, -q2, tol=tol)


@pytest.mark.parametrize("q", [
    [1, 0, 0, 0],
    (1, 0, 0, 0),
    sp.Matrix([1, 0, 0, 0]),
    sp.Matrix([[1, 0, 0, 0]]),
])
def test_quat2rot_accepts_supported_vector_formats(q):
    assert_matrix_equal(quat2rot(q), sp.eye(3))


@pytest.mark.parametrize("q", [
    [],
    [1, 0, 0],
    [1, 0, 0, 0, 0],
    sp.eye(2),
])
def test_quaternion_rejects_invalid_shapes(q):
    with pytest.raises(ValueError, match="4D vector"):
        quat2rot(q)


def test_quat2rot_normalizes_non_unit_input_and_double_coverage():
    q = sp.Matrix([2, 2, 0, 0])

    R1 = quat2rot(q)
    R2 = quat2rot(-3*q)

    assert_matrix_equal(sp.simplify(R1), sp.simplify(R2))
    assert_matrix_equal(R1, rotx(sp.pi/2))


@pytest.mark.parametrize("q", [
    [0, 0, 0, 0],
    [1e-12, 0, 0, 0],
])
def test_quaternion_rejects_zero_or_near_zero_numeric_norm(q):
    with pytest.raises(ValueError):
        quat2rot(q, tol=1e-9)


def test_axa2quat_identity_and_coordinate_axis():
    assert_matrix_equal(
        axa2quat([1, 0, 0], 0),
        sp.Matrix([1, 0, 0, 0]),
    )

    q = axa2quat([0, 0, 1], sp.pi/2)
    expected = sp.Matrix([
        sp.sqrt(2)/2,
        0,
        0,
        sp.sqrt(2)/2,
    ])
    assert_matrix_equal(q, expected)


def test_axa2quat_normalizes_axis_and_supports_degrees():
    q1 = axa2quat([0, 0, 5], 90, deg=True)
    q2 = axa2quat([0, 0, 1], sp.pi/2)
    assert_matrix_equal(q1, q2)


def test_axa2quat_rejects_zero_axis():
    with pytest.raises(ValueError, match="zero vector"):
        axa2quat([0, 0, 0], 1.0)


def test_quat2axa_identity_uses_conventional_x_axis():
    axis, angle = quat2axa([1, 0, 0, 0])
    assert_matrix_equal(axis, sp.Matrix([1, 0, 0]))
    assert sp.simplify(angle) == 0


def test_quat2axa_general_round_trip():
    q = axa2quat([1, -2, 3], 0.9)

    axis, angle = quat2axa(q)

    R1 = quat2rot(q)
    R2 = axa2rot(axis, angle)
    assert_matrix_close(R1, R2, tol=1e-9)


def test_quat2axa_degree_output():
    q = axa2quat([0, 1, 0], 60, deg=True)

    axis, angle = quat2axa(q, deg=True)

    assert_matrix_equal(axis, sp.Matrix([0, 1, 0]))
    assert sp.simplify(angle - 60) == 0


@pytest.mark.parametrize("axis", [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, -2, 3],
])
@pytest.mark.parametrize("angle", [0.2, 1.1, float(sp.pi)-1e-8, float(sp.pi)])
def test_rotation_quaternion_round_trip_numeric(axis, angle):
    R = sp.N(axa2rot(axis, angle))

    q = rot2quat(R)
    R2 = quat2rot(q)

    assert_matrix_close(R2, R, tol=1e-8)
    assert float(sp.N(q[0])) >= -1e-12


@pytest.mark.parametrize("axis", [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
])
def test_rot2quat_exact_pi_reconstructs(axis):
    R = axa2rot(axis, sp.pi)

    q = rot2quat(R)

    assert_matrix_close(quat2rot(q), R, tol=1e-9)


def test_rot2quat_identity_is_canonical():
    q = rot2quat(sp.eye(3))
    assert_matrix_close(q, sp.Matrix([1, 0, 0, 0]), tol=1e-12)


def test_quaternion_matrix_round_trip_from_negative_scalar_input():
    q = sp.Matrix([-sp.sqrt(2)/2, 0, 0, -sp.sqrt(2)/2])

    R = quat2rot(q)
    q2 = rot2quat(R)

    assert_quaternion_equivalent(q, q2)
    assert float(sp.N(q2[0])) >= 0


def test_symbolic_axa2quat_and_quat2rot():
    theta = sp.symbols("theta", real=True)
    q = axa2quat([0, 0, 1], theta)

    expected = sp.Matrix([
        sp.cos(theta/2),
        0,
        0,
        sp.sin(theta/2),
    ])
    assert_matrix_equal(q, expected)

    R = quat2rot(q)
    assert_matrix_equal(sp.trigsimp(R), rotz(theta))


def test_symbolic_rot2quat_reconstructs_rotation():
    theta = sp.symbols("theta", real=True)
    R = rotx(theta)

    q = rot2quat(R)
    R2 = quat2rot(q)

    assert_matrix_equal(sp.trigsimp(R2), R)


def test_symbolic_quaternion_with_indeterminate_norm_is_accepted():
    w, x, y, z = sp.symbols("w x y z", real=True)
    q = sp.Matrix([w, x, y, z])

    R = quat2rot(q)

    assert R.shape == (3, 3)
    assert R.has(w, x, y, z)


@pytest.mark.parametrize("tol", [0, -1e-9])
def test_quaternion_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        quat2rot([1, 0, 0, 0], tol=tol)
    with pytest.raises(ValueError):
        rot2quat(sp.eye(3), tol=tol)
    with pytest.raises(ValueError):
        quat2axa([1, 0, 0, 0], tol=tol)



def test_rotvec2rot_zero_is_exact_identity():
    R = rotvec2rot([0, 0, 0])
    assert_matrix_equal(R, sp.eye(3))


@pytest.mark.parametrize("axis,index", [
    ("x", 0),
    ("y", 1),
    ("z", 2),
])
def test_rotvec2rot_coordinate_axes(axis, index):
    theta = sp.pi / 3
    phi = sp.zeros(3, 1)
    phi[index] = theta

    assert_matrix_equal(rotvec2rot(phi), rot(theta, axis))


def test_rotvec2rot_general_axis_matches_axis_angle():
    axis = sp.Matrix([1, -2, 3])
    axis = axis / axis.norm()
    theta = sp.Rational(7, 10)

    R1 = rotvec2rot(theta * axis)
    R2 = axa2rot(axis, theta)

    assert_matrix_equal(sp.simplify(R1), sp.simplify(R2))


def test_rotvec2rot_small_numeric_angle_is_stable():
    phi = sp.Matrix([1e-10, -2e-10, 3e-10])

    R = rotvec2rot(phi)

    assert is_rotation_matrix(R, tol=1e-9) is True
    assert_matrix_close(R, sp.eye(3) + skew(phi), tol=1e-9)


@pytest.mark.parametrize("angle", [
    float(sp.pi) - 1e-8,
    float(sp.pi),
    1.5 * float(sp.pi),
])
def test_rotvec2rot_large_and_near_pi_magnitudes(angle):
    axis = sp.Matrix([1, 2, -1])
    axis = axis / axis.norm()
    phi = angle * axis

    R = rotvec2rot(phi)

    assert is_rotation_matrix(R, tol=1e-8) is True


def test_rotvec2rot_periodicity_for_same_axis():
    axis = sp.Matrix([1, -2, 3])
    axis = axis / axis.norm()

    R1 = rotvec2rot((sp.pi / 3) * axis)
    R2 = rotvec2rot((sp.pi / 3 + 2 * sp.pi) * axis)

    assert_matrix_equal(sp.trigsimp(R1), sp.trigsimp(R2))


def test_rotvec2rot_symbolic_coordinate_axis():
    theta = sp.symbols("theta", real=True)

    R = rotvec2rot([0, 0, theta])

    assert_matrix_equal(sp.trigsimp(R), rotz(theta))


def test_rot2rotvec_identity_is_zero_vector():
    phi = rot2rotvec(sp.eye(3))
    assert_matrix_equal(phi, sp.zeros(3, 1))


@pytest.mark.parametrize("axis", [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, -2, 3],
])
@pytest.mark.parametrize("angle", [
    0.2,
    1.2,
    float(sp.pi) - 1e-8,
    float(sp.pi),
])
def test_rotation_vector_round_trip_reconstructs(axis, angle):
    R = sp.N(axa2rot(axis, angle))

    phi = rot2rotvec(R)
    R2 = rotvec2rot(phi)

    assert_matrix_close(R2, R, tol=1e-8)
    assert float(sp.N(phi.norm())) <= float(sp.pi) + 1e-8


def test_rot2rotvec_near_identity_returns_zero_with_tolerance():
    R = sp.N(axa2rot([1, 0, 0], 1e-10))

    phi = rot2rotvec(R, tol=1e-9)

    assert_matrix_equal(phi, sp.zeros(3, 1))


def test_rot2rotvec_nonprincipal_input_returns_principal_equivalent():
    axis = sp.Matrix([0, 0, 1])
    original = 3 * sp.pi / 2 * axis

    R = rotvec2rot(original)
    principal = rot2rotvec(R)

    assert float(sp.N(principal.norm())) <= float(sp.pi) + 1e-12
    assert_matrix_equal(
        sp.trigsimp(rotvec2rot(principal)),
        sp.trigsimp(R),
    )


@pytest.mark.parametrize("axis", [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
])
def test_rot2rotvec_exact_pi_reconstructs(axis):
    R = axa2rot(axis, sp.pi)

    phi = rot2rotvec(R)

    assert sp.simplify(phi.norm() - sp.pi) == 0
    assert_matrix_close(rotvec2rot(phi), R, tol=1e-9)


def test_rotation_vector_cross_consistency_with_axis_angle():
    axis = sp.Matrix([1, -2, 3])
    axis = axis / axis.norm()
    theta = sp.Rational(4, 5)
    R = axa2rot(axis, theta)

    recovered_axis, recovered_theta = rot2axa(R)
    phi = rot2rotvec(R)

    assert_matrix_equal(
        sp.simplify(phi),
        sp.simplify(recovered_theta * recovered_axis),
    )


def test_rot2rotvec_symbolic_rotation_reconstructs():
    theta = sp.symbols("theta", real=True)
    R = roty(theta)

    phi = rot2rotvec(R)
    R2 = rotvec2rot(phi)

    assert_matrix_equal(sp.trigsimp(R2), R)


def test_rot2rotvec_rejects_invalid_rotation_matrix():
    with pytest.raises(ValueError, match="rotation matrix"):
        rot2rotvec(sp.diag(1, 1, -1))


@pytest.mark.parametrize("phi", [
    [sp.oo, 0, 0],
    [sp.nan, 0, 0],
    [sp.I, 0, 0],
])
def test_rotvec2rot_rejects_nonfinite_or_nonreal_numeric_inputs(phi):
    with pytest.raises(ValueError, match="finite real"):
        rotvec2rot(phi)
