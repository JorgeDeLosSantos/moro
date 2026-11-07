"""PCHIP-style cubic waypoint velocity estimation (0.7.0 PR-A4)."""

import warnings

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from moro.trajectory import (
    _estimate_shape_preserving_velocities,
    _evaluate_polynomial_segment,
    joint_trajectory,
    joint_waypoint_trajectory,
    position_trajectory,
    position_waypoint_trajectory,
)


def _assert_no_segment_overshoot(values, times):
    """Inspect extrema of actual cubic polynomials, not just output samples."""
    slopes = _estimate_shape_preserving_velocities(values, times)
    for i in range(len(times) - 1):
        duration = times[i + 1] - times[i]
        for j in range(values.shape[1]):
            y0, yf = values[i, j], values[i + 1, j]
            v0, vf = slopes[i, j], slopes[i + 1, j]
            delta = yf - y0
            # Polynomial in local u, with physical-time boundary velocities.
            coefficients = [
                y0,
                duration * v0,
                3 * delta - duration * (2 * v0 + vf),
                -2 * delta + duration * (v0 + vf),
            ]
            polynomial = np.polynomial.Polynomial(coefficients)
            candidates = [0.0, 1.0]
            for root in polynomial.deriv().roots():
                if abs(root.imag) < 1e-12 and 0.0 < root.real < 1.0:
                    candidates.append(float(root.real))
            evaluated = polynomial(candidates)
            scale = max(1.0, abs(y0), abs(yf))
            tolerance = 1e-12 * scale
            assert np.min(evaluated) >= min(y0, yf) - tolerance
            assert np.max(evaluated) <= max(y0, yf) + tolerance


@pytest.mark.parametrize("cartesian", [False, True])
def test_shape_preserving_matches_scipy_interior_slopes(cartesian):
    times = np.array([0.0, 0.7, 2.1, 2.9, 5.2])
    values = np.array([
        [0.0, 6.0, 0.0],
        [0.5, 5.0, 1.0],
        [1.3, 3.0, 2.0],
        [1.8, 2.0, 1.5],
        [3.0, 0.0, 0.0],
    ])
    estimated = _estimate_shape_preserving_velocities(values, times)
    reference = PchipInterpolator(times, values, axis=0).derivative()(times)

    np.testing.assert_allclose(
        estimated[1:-1], reference[1:-1], atol=1e-12, rtol=1e-12
    )
    np.testing.assert_array_equal(estimated[[0, -1]], 0.0)

    generator = (
        position_waypoint_trajectory if cartesian
        else joint_waypoint_trajectory
    )
    result = generator(
        values, times, method="cubic", derivative_policy="shape_preserving",
        samples_per_segment=11,
    )
    derivative_samples = result.v if cartesian else result.qd
    np.testing.assert_allclose(
        derivative_samples[::10], estimated, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_array_equal(result.t[::10], times)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 3.0],
        [0.0, 1.0, -1.0, 2.0, -2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ],
)
def test_exact_cubic_extrema_are_bounded_by_segment_waypoints(values):
    times = np.array([0.0, 0.7, 1.9, 3.1, 5.0])
    _assert_no_segment_overshoot(
        np.asarray(values, dtype=float)[:, None], times
    )


def test_coordinate_wise_reversals_plateaus_and_mixed_data():
    times = np.array([0.0, 0.5, 2.2, 3.0, 5.5])
    values = np.array([
        [0.0, 6.0, 0.0, 1.0],
        [1.0, 5.0, 1.0, 1.0],
        [1.7, 3.0, 2.0, 1.0],
        [1.8, 2.0, 1.5, 3.0],
        [5.0, 0.0, 0.5, 3.0],
    ])
    velocities = _estimate_shape_preserving_velocities(values, times)
    assert velocities[2, 2] == 0.0  # Turning point in coordinate 3
    assert velocities[1, 3] == 0.0  # Plateau in coordinate 4
    assert velocities[2, 3] == 0.0
    assert velocities[3, 3] == 0.0
    assert velocities[2, 0] > 0
    assert velocities[2, 1] < 0
    _assert_no_segment_overshoot(values, times)

    trajectory = joint_waypoint_trajectory(
        values, times, method="cubic",
        derivative_policy="shape_preserving", samples_per_segment=25,
    )
    for k in range(len(times) - 1):
        samples = trajectory.q[24*k:24*(k+1)+1]
        for col in range(values.shape[1]):
            assert np.min(samples[:, col]) >= (
                min(values[k, col], values[k+1, col]) - 1e-12
            )
            assert np.max(samples[:, col]) <= (
                max(values[k, col], values[k+1, col]) + 1e-12
            )


def test_shape_preserving_c1_continuity_using_two_sided_polynomials():
    values = np.array([
        [0.0, 2.0],
        [1.0, 3.0],
        [1.7, 2.5],
        [2.2, 5.0],
    ])
    times = np.array([-2.0, -1.5, 0.1, 3.0])
    velocities = _estimate_shape_preserving_velocities(values, times)
    for i in range(1, len(times) - 1):
        left = _evaluate_polynomial_segment(
            values[i-1], values[i], times[i-1], times[i], [times[i]],
            method="cubic", v0=velocities[i-1], vf=velocities[i],
        )
        right = _evaluate_polynomial_segment(
            values[i], values[i+1], times[i], times[i+1], [times[i]],
            method="cubic", v0=velocities[i], vf=velocities[i+1],
        )
        np.testing.assert_allclose(left[1], right[1], atol=1e-12)
        np.testing.assert_allclose(left[2], right[2], atol=1e-12)


@pytest.mark.parametrize("cartesian", [False, True])
def test_two_waypoint_shape_preserving_matches_resting_cubic(cartesian):
    times = [1.0, 1.25, 2.0, 4.0]
    if cartesian:
        points = [[0.0, 0.5, 2.0], [1.0, -1.0, 3.0]]
        actual = position_waypoint_trajectory(
            points, [1.0, 4.0], method="cubic",
            derivative_policy="shape_preserving", sample_times=times,
        )
        reference = position_trajectory(
            points[0], points[1], times, method="cubic"
        )
        for a, b in zip(
            (actual.p, actual.v, actual.a),
            (reference.p, reference.v, reference.a),
        ):
            np.testing.assert_allclose(a, b, atol=1e-12)
    else:
        points = [0.0, 1.0]
        actual = joint_waypoint_trajectory(
            points, [1.0, 4.0], method="cubic",
            derivative_policy="shape_preserving", sample_times=times,
        )
        reference = joint_trajectory(0.0, 1.0, times, method="cubic")
        for a, b in zip(
            (actual.q, actual.qd, actual.qdd),
            (reference.q, reference.qd, reference.qdd),
        ):
            np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize(
    "method,velocities,accelerations",
    [
        ("linear", None, None),
        ("quintic", None, None),
        ("cubic", [0.0, 0.0, 0.0], None),
        ("cubic", None, [0.0, 0.0, 0.0]),
        ("cubic", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_shape_preserving_rejects_incompatible_inputs(
    method, velocities, accelerations
):
    with pytest.raises(ValueError, match="shape_preserving"):
        joint_waypoint_trajectory(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            method=method,
            derivative_policy="shape_preserving",
            velocities=velocities,
            accelerations=accelerations,
        )


def test_shape_preserving_case_insensitive_and_sparse_samples():
    times = [0.0, 0.8, 2.1, 4.0]
    samples = [0.0, 0.2, 1.1, 1.5, 3.0, 4.0]
    out = joint_waypoint_trajectory(
        [0.0, 0.5, 2.0, 1.0], times, method="CUBIC",
        derivative_policy="SHAPE_PRESERVING", sample_times=samples,
    )
    assert out.method == "cubic"
    np.testing.assert_array_equal(out.t, samples)
    assert 0.8 not in out.t
    assert 2.1 not in out.t


def test_shape_preserving_avoids_overflow_for_tiny_secants():
    values = np.array([0.0, 1e-300, 2e-300, 3e-300])[:, None]
    times = np.array([0.0, 1.0, 2.0, 3.0])
    result = _estimate_shape_preserving_velocities(values, times)
    assert np.all(result[1:-1, 0] > 0)
    np.testing.assert_allclose(result[1:-1, 0], [1e-300, 1e-300],
                               atol=0, rtol=1e-12)


def test_shape_preserving_large_durations_remain_finite():
    times = np.array([0.0, 1e200, 2e200, 3e200])
    values = np.array([0.0, 1e-100, 3e-100, 4e-100])[:, None]
    result = _estimate_shape_preserving_velocities(values, times)
    assert np.isfinite(result).all()
    assert np.all(result[1:-1] > 0)


def test_shape_preserving_rejects_nonfinite_secants_without_warnings():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match="finite waypoint secant"):
            joint_waypoint_trajectory(
                [-1e308, 1e308, 0.0], [0.0, 1.0, 2.0],
                method="cubic", derivative_policy="shape_preserving",
            )
    assert not any(issubclass(w.category, RuntimeWarning) for w in recorded)


def test_default_zero_policy_is_unchanged():
    p = [0.0, 1.0, 3.0]
    t = [0.0, 1.0, 2.0]
    a = joint_waypoint_trajectory(p, t, method="cubic")
    b = joint_waypoint_trajectory(
        p, t, method="cubic", derivative_policy="zero",
    )
    np.testing.assert_allclose(a.q, b.q)
    np.testing.assert_allclose(a.qd, b.qd)
    np.testing.assert_allclose(a.qdd, b.qdd)
    np.testing.assert_allclose(a.qd[[0, 50, 100]], 0.0, atol=1e-12)
