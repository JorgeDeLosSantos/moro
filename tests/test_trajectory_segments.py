"""Tests for the private polynomial-segment evaluator introduced in 0.7.0 A1.

The public waypoint API is intentionally not part of this increment.
"""

import numpy as np
import pytest

from moro.trajectory import (
    _evaluate_polynomial_segment,
    joint_trajectory,
    position_trajectory,
)


@pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
def test_segment_matches_joint_point_to_point_on_full_grid(method):
    t = np.array([2.0, 2.1, 2.45, 3.0, 4.0])
    q0 = np.array([0.2, -0.5])
    qf = np.array([0.8, 1.3])
    kwargs = {}
    if method in ("cubic", "quintic"):
        kwargs.update(v0=np.array([0.1, -0.2]), vf=np.array([0.3, 0.4]))
    if method == "quintic":
        kwargs.update(a0=np.array([0.03, 0.1]), af=np.array([-0.2, 0.05]))

    actual = _evaluate_polynomial_segment(
        q0, qf, t[0], t[-1], t, method=method, **kwargs
    )
    public_kwargs = {}
    if "v0" in kwargs:
        public_kwargs.update(qd0=kwargs["v0"], qdf=kwargs["vf"])
    if "a0" in kwargs:
        public_kwargs.update(qdd0=kwargs["a0"], qddf=kwargs["af"])
    expected = joint_trajectory(q0, qf, t, method=method, **public_kwargs)

    for got, reference in zip(actual, (expected.t, expected.q, expected.qd, expected.qdd)):
        np.testing.assert_allclose(got, reference)


@pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
def test_partial_samples_use_true_segment_bounds(method):
    t0, tf = 2.0, 6.0
    x0 = np.array([0.0, -1.0])
    xf = np.array([4.0, 3.0])
    samples = np.array([3.0, 4.5])
    actual_t, x, xd, xdd = _evaluate_polynomial_segment(
        x0, xf, t0, tf, samples, method=method
    )
    reference = joint_trajectory(x0, xf, [t0, *samples, tf], method=method)

    np.testing.assert_array_equal(actual_t, samples)
    np.testing.assert_allclose(x, reference.q[1:-1])
    np.testing.assert_allclose(xd, reference.qd[1:-1])
    np.testing.assert_allclose(xdd, reference.qdd[1:-1])


@pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
def test_single_sample_grid_and_empty_grid(method):
    sample = np.array([2.0])
    for samples in (sample, np.array([])):
        t, x, xd, xdd = _evaluate_polynomial_segment(
            [0.0, 1.0], [3.0, -2.0], 1.0, 4.0, samples, method=method
        )
        assert t.shape == (len(samples),)
        for values in (x, xd, xdd):
            assert values.shape == (len(samples), 2)
            assert np.isfinite(values).all()

    reference = joint_trajectory([0.0, 1.0], [3.0, -2.0], [1.0, 2.0, 4.0], method=method)
    got = _evaluate_polynomial_segment(
        [0.0, 1.0], [3.0, -2.0], 1.0, 4.0, sample, method=method
    )
    np.testing.assert_allclose(got[1], reference.q[1:2])
    np.testing.assert_allclose(got[2], reference.qd[1:2])
    np.testing.assert_allclose(got[3], reference.qdd[1:2])


@pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
def test_nonzero_boundaries_are_preserved_with_partial_samples(method):
    t0, tf = -2.0, 3.0
    x0, xf = np.array([1.0, 0.0]), np.array([3.0, -4.0])
    kwargs = dict(v0=np.array([0.7, -0.1]), vf=np.array([-0.3, 0.5]))
    if method == "quintic":
        kwargs.update(a0=np.array([0.8, 0.1]), af=np.array([-0.2, -0.4]))
    if method == "linear":
        kwargs.clear()

    result = _evaluate_polynomial_segment(
        x0, xf, t0, tf, [0.0, 1.5], method=method, **kwargs
    )
    public_kwargs = {}
    if "v0" in kwargs:
        public_kwargs.update(qd0=kwargs["v0"], qdf=kwargs["vf"])
    if "a0" in kwargs:
        public_kwargs.update(qdd0=kwargs["a0"], qddf=kwargs["af"])
    expected = joint_trajectory(
        x0, xf, [t0, 0.0, 1.5, tf], method=method, **public_kwargs
    )
    np.testing.assert_allclose(result[1], expected.q[1:-1])
    np.testing.assert_allclose(result[2], expected.qd[1:-1])
    np.testing.assert_allclose(result[3], expected.qdd[1:-1])


@pytest.mark.parametrize(
    "t0,tf,samples",
    [
        (1.0, 1.0, [1.0]),
        (2.0, 1.0, [1.5]),
        (np.nan, 2.0, [1.5]),
        (1.0, np.inf, [1.5]),
        (0.0, 1.0, [-0.1]),
        (0.0, 1.0, [1.1]),
        (0.0, 1.0, [0.6, 0.5]),
        (0.0, 1.0, [0.4, 0.4]),
        (0.0, 1.0, [np.nan]),
        (0.0, 1.0, [[0.5]]),
    ],
)
def test_invalid_segment_times_are_rejected(t0, tf, samples):
    with pytest.raises(ValueError):
        _evaluate_polynomial_segment(
            [0.0], [1.0], t0, tf, samples, method="quintic"
        )


def test_cartesian_public_api_stays_compatible():
    times = np.array([1.0, 1.7, 2.1, 3.5])
    p0, pf = np.array([0.0, 0.5, 1.0]), np.array([2.0, -0.8, 1.2])
    derivatives = dict(
        v0=np.array([0.3, 0.1, -0.2]),
        vf=np.array([0.0, 0.3, 0.2]),
        a0=np.array([0.1, 0.2, -0.3]),
        af=np.array([-0.2, 0.0, 0.2]),
    )
    expected = position_trajectory(p0, pf, times, method="quintic", **derivatives)
    t, p, v, a = _evaluate_polynomial_segment(
        p0, pf, times[0], times[-1], times,
        method="quintic", **derivatives
    )
    for actual, ref in zip((t, p, v, a), (expected.t, expected.p, expected.v, expected.a)):
        np.testing.assert_allclose(actual, ref)
