"""Tests for sampled workspace analysis and visualization."""

import numpy as np
import pytest
import sympy as sp
import matplotlib.pyplot as plt

import moro.workspace as workspace_module
from moro.abc import q1, q2, q3
from moro.core import Robot
from moro.workspace import Workspace, sample_workspace
from moro.visualization import plot_workspace


def planar_2r():
    robot = Robot((1.0, 0, 0, q1, "r"), (0.8, 0, 0, q2, "r"))
    robot.joint_limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2)]
    return robot


def spatial_3r():
    robot = Robot(
        (0.4, sp.pi / 2, 0.2, q1, "r"),
        (0.3, -sp.pi / 2, 0.0, q2, "r"),
        (0.2, 0.0, 0.1, q3, "r"),
    )
    robot.joint_limits = [(-1.0, 1.0), (-1.2, 1.2), (-0.8, 0.8)]
    return robot


def prismatic_1dof():
    robot = Robot((0, 0, q1, 0, "p"))
    robot.joint_limits = [(0.1, 0.9)]
    return robot


def mixed_rp():
    robot = Robot((1.0, 0, 0, q1, "r"), (0.0, 0, q2, 0, "p"))
    robot.joint_limits = [(-np.pi, np.pi), (0.0, 0.5)]
    return robot


class TestWorkspace:
    def test_properties_bounds_and_defensive_copying(self):
        points = np.array([[0.0, -1.0, 2.0], [2.0, 3.0, -1.0]])
        configs = np.array([[0.0, 0.5], [1.0, 1.5]])
        ws = Workspace(points, configs, ((-1, 2), (0, 2)), seed=12)

        points[:] = -100
        configs[:] = -100

        assert ws.samples == 2
        assert ws.dof == 2
        assert ws.seed == 12
        assert ws.joint_limits == ((-1.0, 2.0), (0.0, 2.0))
        assert ws.bounds == ((0.0, 2.0), (-1.0, 3.0), (-1.0, 2.0))
        assert "Workspace(samples=2, dof=2, seed=12)" == repr(ws)

        ws.points[0, 0] = -2.0
        assert ws.bounds[0] == (-2.0, 2.0)

    @pytest.mark.parametrize(
        "points, configs, limits",
        [
            ([0.0, 0.0, 0.0], [[0.0]], ((-1, 1),)),
            ([[0.0, 0.0]], [[0.0]], ((-1, 1),)),
            ([[0.0, 0.0, 0.0]], [[0.0], [0.1]], ((-1, 1),)),
            ([[np.nan, 0.0, 0.0]], [[0.0]], ((-1, 1),)),
            ([[0.0, 0.0, 0.0]], [[np.inf]], ((-1, 1),)),
            ([[0.0, 0.0, 0.0]], [[2.0]], ((-1, 1),)),
            ([[0.0, 0.0, 0.0]], [[0.0]], ((1, 1),)),
        ],
    )
    def test_invalid_workspace_invariants(self, points, configs, limits):
        with pytest.raises(ValueError):
            Workspace(points, configs, limits)


class TestSamplingValidation:
    @pytest.mark.parametrize("samples", [0, -1, 1.5, True, "10"])
    def test_invalid_sample_count(self, samples):
        with pytest.raises(ValueError):
            sample_workspace(planar_2r(), samples=samples)

    @pytest.mark.parametrize("seed", [True, 1.5, "1"])
    def test_invalid_seed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            sample_workspace(planar_2r(), samples=2, seed=seed)

    @pytest.mark.parametrize(
        "limits",
        [
            [(-1.0, 1.0)],
            [(-1.0, 1.0), None],
            [(-1.0, 1.0), (0.0,)],
            [(-1.0, 1.0), (0.0, np.inf)],
            [(-1.0, 1.0), (np.nan, 1.0)],
            [(-1.0, 1.0), (1.0, 1.0)],
            [(-1.0, 1.0), (2.0, 1.0)],
            [(-1.0, 1.0), (False, 1.0)],
        ],
    )
    def test_invalid_joint_limits(self, limits):
        with pytest.raises(ValueError):
            sample_workspace(planar_2r(), samples=2, joint_limits=limits)

    def test_samples_one_and_explicit_limit_precedence(self):
        robot = planar_2r()
        original = robot.joint_limits
        override = [(-0.2, 0.2), (-0.3, 0.3)]

        ws = sample_workspace(robot, samples=1, joint_limits=override, seed=4)

        assert ws.samples == 1
        assert ws.points.shape == (1, 3)
        assert ws.configurations.shape == (1, 2)
        assert ws.joint_limits == ((-0.2, 0.2), (-0.3, 0.3))
        assert robot.joint_limits == original


class TestSamplingBehavior:
    def test_seed_reproducibility_and_global_rng_independence(self):
        robot = planar_2r()
        a = sample_workspace(robot, samples=30, seed=123)
        b = sample_workspace(robot, samples=30, seed=123)
        np.testing.assert_allclose(a.configurations, b.configurations)
        np.testing.assert_allclose(a.points, b.points)

        np.random.seed(4321)
        expected = np.random.random(3)
        np.random.seed(4321)
        sample_workspace(robot, samples=5, seed=7)
        observed = np.random.random(3)
        np.testing.assert_allclose(observed, expected)

    def test_configurations_inside_limits_and_fk_correspondence(self):
        robot = planar_2r()
        ws = sample_workspace(robot, samples=12, seed=5)
        limits = np.asarray(ws.joint_limits)

        assert np.all(ws.configurations >= limits[:, 0])
        assert np.all(ws.configurations <= limits[:, 1])

        for index in [0, 3, 7, 11]:
            q = ws.configurations[index]
            expected = np.asarray(
                robot.T[:3, 3].subs(dict(zip(robot.qs, q))),
                dtype=float,
            ).reshape(3)
            np.testing.assert_allclose(ws.points[index], expected, atol=1e-12)

    def test_planar_2r_sanity(self):
        ws = sample_workspace(planar_2r(), samples=200, seed=2)
        np.testing.assert_allclose(ws.points[:, 2], 0.0, atol=1e-12)
        radius = np.linalg.norm(ws.points[:, :2], axis=1)
        assert np.max(radius) <= 1.8 + 1e-12

    def test_prismatic_and_mixed_robots(self):
        prismatic = sample_workspace(prismatic_1dof(), samples=50, seed=9)
        np.testing.assert_allclose(prismatic.points[:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            prismatic.points[:, 2],
            prismatic.configurations[:, 0],
            atol=1e-12,
        )

        mixed = sample_workspace(mixed_rp(), samples=50, seed=10)
        assert mixed.configurations.shape == (50, 2)
        np.testing.assert_allclose(
            mixed.points[:, 2],
            mixed.configurations[:, 1],
            atol=1e-12,
        )

    def test_spatial_robot_has_xyz_variation(self):
        ws = sample_workspace(spatial_3r(), samples=100, seed=3)
        spans = np.array([high - low for low, high in ws.bounds])
        assert np.all(spans > 1e-6)


class TestSymbolicParametersAndFailures:
    def test_symbolic_geometry_parameters_without_robot_mutation(self):
        l1, l2 = sp.symbols("l1 l2", positive=True)
        robot = Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))
        robot.joint_limits = [(-1.0, 1.0), (-1.0, 1.0)]

        ws = sample_workspace(
            robot,
            samples=10,
            parameters={l1: 1.0, l2: 0.5},
            seed=12,
        )

        assert ws.points.shape == (10, 3)
        assert robot.T.has(l1)
        assert robot.T.has(l2)

    def test_missing_parameter_fails_before_rng_generation(self, monkeypatch):
        length = sp.symbols("length", positive=True)
        robot = Robot((length, 0, 0, q1, "r"))
        robot.joint_limits = [(-1.0, 1.0)]
        called = {"rng": False}
        original = np.random.default_rng

        def tracking_rng(*args, **kwargs):
            called["rng"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(workspace_module.np.random, "default_rng", tracking_rng)
        with pytest.raises(ValueError, match="length"):
            sample_workspace(robot, samples=3, seed=1)
        assert called["rng"] is False

    def test_numerical_fk_failure_aborts_full_sampling(self, monkeypatch):
        robot = planar_2r()
        monkeypatch.setattr(
            workspace_module,
            "_prepare_position_model",
            lambda robot, parameters=None: lambda *q: [np.nan, 0.0, 0.0],
        )
        with pytest.raises(ValueError, match="sample 0"):
            sample_workspace(robot, samples=5, seed=1)


class TestWorkspaceVisualization:
    @staticmethod
    def ws(points):
        points = np.asarray(points, dtype=float)
        configs = np.linspace(-0.5, 0.5, points.shape[0])[:, None]
        return Workspace(points, configs, ((-1.0, 1.0),), seed=1)

    def test_auto_projection_for_xy_xz_yz_and_3d(self):
        cases = [
            ([[0, 0, 0], [1, 1, 0]], ("X", "Y"), False),
            ([[0, 0, 0], [1, 0, 1]], ("X", "Z"), False),
            ([[0, 0, 0], [0, 1, 1]], ("Y", "Z"), False),
            ([[0, 0, 0], [1, 1, 1]], ("X", "Y"), True),
        ]
        for points, labels, expect_3d in cases:
            fig, ax = plot_workspace(self.ws(points), projection="auto")
            assert (ax.get_xlabel(), ax.get_ylabel()) == labels
            assert (ax.name == "3d") is expect_3d
            plt.close(fig)

    @pytest.mark.parametrize(
        "projection, labels",
        [("xy", ("X", "Y")), ("xz", ("X", "Z")), ("yz", ("Y", "Z"))],
    )
    def test_explicit_2d_projections_and_point_count(self, projection, labels):
        ws = self.ws([[0, 0, 0], [1, 2, 3], [2, 1, 0]])
        fig, ax = plot_workspace(ws, projection=projection)
        assert (ax.get_xlabel(), ax.get_ylabel()) == labels
        assert len(ax.collections[0].get_offsets()) == ws.samples
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_explicit_3d_and_existing_axis_validation(self):
        ws = self.ws([[0, 0, 0], [1, 2, 3]])
        fig, ax = plot_workspace(ws, projection="3d")
        assert ax.name == "3d"
        assert ax.get_zlabel() == "Z"
        plt.close(fig)

        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_workspace(ws, projection="xy", ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        with pytest.raises(ValueError, match="3D axis"):
            plot_workspace(ws, projection="3d", ax=ax)
        plt.close(fig)

    @pytest.mark.parametrize("projection", ["xyz", "plane", 3])
    def test_invalid_projection(self, projection):
        with pytest.raises(ValueError, match="projection"):
            plot_workspace(self.ws([[0, 0, 0], [1, 1, 0]]), projection=projection)
