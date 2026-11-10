"""W-03 regression tests for alternate sampled-workspace distributions."""

import numpy as np
import pytest
import sympy as sp

from moro.abc import q1, q2
from moro.core import Robot
from moro.workspace import Workspace, sample_workspace


def rr():
    robot = Robot((1.0, 0, 0, q1, "r"), (0.8, 0, 0, q2, "r"))
    robot.joint_limits = [(-1.0, 1.0), (-0.5, 0.5)]
    return robot


@pytest.mark.parametrize("method", ["random", "sobol", "latin_hypercube"])
@pytest.mark.parametrize("samples", [1, 7, 8, 17])
def test_stochastic_shapes_limits_reproducibility_and_fk(method, samples):
    robot = rr()
    a = sample_workspace(robot, method=method, samples=samples, seed=17)
    b = sample_workspace(robot, method=method, samples=samples, seed=17)
    assert a.points.shape == (samples, 3)
    assert a.configurations.shape == (samples, 2)
    assert a.method == method and a.grid_shape is None and a.seed == 17
    np.testing.assert_array_equal(a.configurations, b.configurations)
    np.testing.assert_allclose(a.points, b.points)
    limits = np.asarray(a.joint_limits)
    assert np.all(a.configurations >= limits[:, 0])
    assert np.all(a.configurations <= limits[:, 1])
    for k in {0, samples - 1}:
        expected = np.asarray(
            robot.T[:3, 3].subs(dict(zip(robot.qs, a.configurations[k]))),
            dtype=float,
        ).reshape(3)
        np.testing.assert_allclose(a.points[k], expected, atol=1e-12)


def test_legacy_random_seed_sequence_and_effective_default():
    robot = rr()
    expected = np.random.default_rng(11).uniform(
        low=[-1.0, -0.5], high=[1.0, 0.5], size=(13, 2),
    )
    a = sample_workspace(robot, samples=13, seed=11)
    np.testing.assert_array_equal(a.configurations, expected)
    assert a.method == "random"
    assert sample_workspace(robot, seed=1).samples == 1000


@pytest.mark.parametrize("method", ["sobol", "latin_hypercube"])
def test_qmc_global_rng_independence(method):
    robot = rr()
    np.random.seed(213)
    baseline = np.random.random(5)
    np.random.seed(213)
    sample_workspace(robot, method=method, samples=5, seed=11)
    np.testing.assert_array_equal(np.random.random(5), baseline)


def test_grid_order_endpoints_midpoint_and_metadata():
    robot = rr()
    result = sample_workspace(robot, method="grid", grid_shape=(3, 2))
    np.testing.assert_array_equal(
        result.configurations,
        [[-1.0, -0.5], [-1.0, 0.5],
         [0.0, -0.5], [0.0, 0.5],
         [1.0, -0.5], [1.0, 0.5]],
    )
    assert result.method == "grid"
    assert result.grid_shape == (3, 2)
    assert result.seed is None and result.samples == 6
    midpoint = sample_workspace(robot, method="grid", grid_shape=(1, 1))
    np.testing.assert_array_equal(midpoint.configurations, [[0.0, 0.0]])


@pytest.mark.parametrize("method,kwargs", [
    ("unknown", {}),
    (3, {}),
    ("grid", {}),
    ("grid", {"samples": 5, "grid_shape": (2, 3)}),
    ("grid", {"seed": 2, "grid_shape": (2, 3)}),
    ("grid", {"grid_shape": (0, 3)}),
    ("grid", {"grid_shape": (True, 3)}),
    ("grid", {"grid_shape": (2, 1.5)}),
    ("grid", {"grid_shape": (2,)}),
    ("grid", {"grid_shape": (1001, 1001)}),
    ("random", {"grid_shape": (2, 3)}),
    ("sobol", {"grid_shape": (2, 3)}),
    ("latin_hypercube", {"grid_shape": (2, 3)}),
    ("sobol", {"samples": 0}),
    ("random", {"samples": 1_000_001}),
])
def test_invalid_sampling_contract(method, kwargs):
    with pytest.raises(ValueError):
        sample_workspace(rr(), method=method, **kwargs)


def test_direct_workspace_backward_compatibility_and_provenance_checks():
    points = [[0.0, 0.0, 0.0]]
    configs = [[0.0]]
    a = Workspace(points, configs, ((-1, 1),), 3)
    assert a.method is None and a.grid_shape is None
    assert repr(a) == "Workspace(samples=1, dof=1, seed=3)"
    with pytest.raises(ValueError, match="grid_shape"):
        Workspace(points, configs, ((-1, 1),), method="grid")
    with pytest.raises(ValueError, match="product"):
        Workspace(points, configs, ((-1, 1),),
                  method="grid", grid_shape=(2,))


def test_symbolic_model_parameter_and_no_mutation():
    length = sp.Symbol("length", positive=True)
    robot = Robot((length, 0, 0, q1, "r"))
    robot.joint_limits = [(-1.0, 1.0)]
    original = tuple(robot.joint_limits)
    for method, kwargs in [
        ("sobol", {"samples": 8, "seed": 4}),
        ("latin_hypercube", {"samples": 8, "seed": 4}),
        ("grid", {"grid_shape": (8,)}),
    ]:
        result = sample_workspace(
            robot, method=method, parameters={length: 1.5}, **kwargs
        )
        assert result.points.shape == (8, 3)
        assert tuple(robot.joint_limits) == original
        assert robot.T.has(length)
