import sympy as sp

from moro.transformations import axa2rot, rot2axa


def assert_matrix_equal(a, b):
    diff = a - b
    assert all(sp.simplify(v) == 0 for v in diff)


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