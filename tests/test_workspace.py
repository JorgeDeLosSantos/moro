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


def make_planar_2r():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.8, 0, 0, q2, "r"),
    )
    robot.joint_limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2)]
    return robot


def make_spatial_3r():
    robot = Robot(
        (0.4, sp.pi / 2, 0.2, q1, "r"),
        (0.3, -sp.pi / 2, 0.0, q2, "r"),
        (0.2, 0.0, 0.1, q3, "r"),
    )
    robot.joint_limits = [(-1.0, 1.0), (-1.2, 1.2), (-0.8, 0.8)]
    return robot


def make_prismatic_1dof():
    robot = Robot((0, 0, q1, 0, "p"))
    robot.joint_limits = [(0.1, 0.9)]
    return robot


def make_mixed_rp():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (0.0, 0, q2, 0, "p"),
    )
    robot.joint_limits = [(-np.pi, np.pi), (0.0, 0.5)]
    return robot


class TestWorkspaceResult:
    def test_valid_result_properties_and_bounds(self):
        ws = Workspace(
            points=[
                [0.0, -1.0, 2.0],
                [2.0, 3.0, -1.0],
            ],
            configurations=[
                [0.0, 0.5],
                [1.0, 1.5],
            ],
            joint_limits=((-1.0, 2.0), (0.0, 2.0)),
            seed=12,
        )

        assert ws.samples == 2
        assert ws.dof == 2
        assert ws.seed == 12
        assert ws.joint_limits == ((-1.0, 2.0), (0.0, 2.0))
        assert ws.bounds == ((0.0, 2.0), (-1.0, 3.0), (-1.0, 2.0))
        assert "Workspace(samples=2, dof=2, seed=12)" == repr(ws)

    def test_defensive_copying_and_dynamic_bounds(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        configs = np.array([[0.0], [0.5]])
        ws = Workspace(points, configs, ((-1.0, 1.0),), seed=None)

        points[:] = -100.0
        configs[:] = -100.0

        np.testing.assert_allclose(ws.points[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ws.configurations[:, 0], [0.0, 0.5])

        ws.points[0, 0] = -2.0
        assert ws.bounds[0] == (-2.0, 1.0)

    @pytest.mark.parametrize(
        "points",
        [
            [0.0, 0.0, 0.0],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [],
        ],
    )
    def test_rejects_invalid_point_shapes(self, points):
        with pytest.raises(ValueError):
            Workspace(points, [[0.0]], ((-1.0, 1.0),))

    def test_rejects_mismatched_sample_counts(self):
        with pytest.raises(ValueError, match="same number"):
            Workspace(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0]],
                ((-1.0, 1.0),),
            )

    @pytest.mark.parametrize(
        "points, configs",
        [
            ([[np.nan, 0.0, 0.0]], [[0.0]]),
            ([[np.inf, 0.0, 0.0]], [[0.0]]),
            ([[0.0, 0.0, 0.0]], [[np.nan]]),
            ([[0.0, 0.0, 0.0]], [[np.inf]]),
            ([[1j, 0.0, 0.0]], [[0.0]]),
        ],
    )
    def test_rejects_nonfinite_or_complex_data(self, points, configs):
        with pytest.raises(ValueError):
            Workspace(points, configs, ((-1.0, 1.0),))

    def test_rejects_configuration_outside_stored_limits(self):
        with pytest.raises(ValueError, match="stored joint_limits"):
            Workspace(
                [[0.0, 0.0, 0.0]],
                [[2.0]],
                ((-1.0, 1.0),),
            )


class TestWorkspaceValidation:
    @pytest.mark.parametrize("samples", [0, -1, 1.5, True, "10"])
    def test_rejects_invalid_sample_counts(self, samples):
        with pytest.raises(ValueError):
            sample_workspace(make_planar_2r(), samples=samples)

    def test_samples_one_is_valid(self):
        ws = sample_workspace(make_planar_2r(), samples=1, seed=1)
        assert ws.samples == 1
        assert ws.points.shape == (1, 3)
        assert ws.configurations.shape == (1, 2)

    @pytest.mark.parametrize("seed", [True, 1.5, "1"])
    def test_rejects_invalid_seed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            sample_workspace(make_planar_2r(), samples=2, seed=seed)

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
    def test_rejects_invalid_joint_limits(self, limits):
        with pytest.raises(ValueError):
            sample_workspace(
                make_planar_2r(),
                samples=2,
                joint_limits=limits,
            )

    def test_explicit_limits_override_without_mutating_robot(self):
        robot = make_planar_2r()
        original = robot.joint_limits
        override = [(-0.2, 0.2), (-0.3, 0.3)]

        ws = sample_workspace(
            robot,
            samples=20,
            joint_limits=override,
            seed=4,
        )

        assert ws.joint_limits == ((-0.2, 0.2), (-0.3, 0.3))
        assert robot.joint_limits == original
        assert np.all(ws.configurations[:, 0] >= -0.2)
        assert np.all(ws.configurations[:, 0] <= 0.2)
        assert np.all(ws.configurations[:, 1] >= -0.3)
        assert np.all(ws.configurations[:, 1] <= 0.3)


class TestWorkspaceSampling:
    def test_same_seed_reproduces_configurations_and_points(self):
        robot = make_planar_2r()
        a = sample_workspace(robot, samples=50, seed=123)
        b = sample_workspace(robot, samples=50, seed=123)

        np.testing.assert_allclose(a.configurations, b.configurations)
        np.testing.assert_allclose(a.points, b.points)

    def test_sampling_does_not_use_global_numpy_rng_state(self):
        robot = make_planar_2r()
        np.random.seed(1234)
        expected = np.random.random(3)

        np.random.seed(1234)
        sample_workspace(robot, samples=5, seed=7)
        observed = np.random.random(3)

        np.testing.assert_allclose(observed, expected)

    def test_every_configuration_lies_inside_effective_limits(self):
        robot = make_planar_2r()
        ws = sample_workspace(robot, samples=100, seed=8)
        limits = np.asarray(ws.joint_limits)

        assert np.all(ws.configurations >= limits[:, 0])
        assert np.all(ws.configurations <= limits[:, 1])

    def test_point_configuration_correspondence(self):
        robot = make_planar_2r()
        ws = sample_workspace(robot, samples=12, seed=5)

        for index in [0, 3, 7, 11]:
            q = ws.configurations[index]
            expected = np.asarray(
                robot.T[:3, 3].subs(dict(zip(robot.qs, q))),
                dtype=float,
            ).reshape(3)
            np.testing.assert_allclose(ws.points[index], expected, atol=1e-12)

    def test_planar_2r_sanity(self):
        robot = make_planar_2r()
        ws = sample_workspace(robot, samples=200, seed=2)

        np.testing.assert_allclose(ws.points[:, 2], 0.0, atol=1e-12)
        radius = np.linalg.norm(ws.points[:, :2], axis=1)
        assert np.max(radius) <= pytest.approx(1.8, abs=1e-12)

    def test_prismatic_workspace_tracks_prismatic_range(self):
        robot = make_prismatic_1dof()
        ws = sample_workspace(robot, samples=100, seed=9)

        np.testing.assert_allclose(ws.points[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(ws.points[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            ws.points[:, 2],
            ws.configurations[:, 0],
            atol=1e-12,
        )

    def test_mixed_revolute_prismatic_robot(self):
        robot = make_mixed_rp()
        ws = sample_workspace(robot, samples=50, seed=10)

        assert ws.configurations.shape == (50, 2)
        assert ws.points.shape == (50, 3)
        np.testing.assert_allclose(
            ws.points[:, 2],
            ws.configurations[:, 1],
            atol=1e-12,
        )

    def test_spatial_robot_produces_three_dimensional_variation(self):
        ws = sample_workspace(make_spatial_3r(), samples=100, seed=3)
        spans = np.array([high - low for low, high in ws.bounds])
        assert np.all(spans > 1e-6)


class TestWorkspaceParameters:
    def test_symbolic_geometry_parameters_and_no_robot_mutation(self):
        l1, l2 = sp.symbols("l1 l2", positive=True)
        robot = Robot(
            (l1, 0, 0, q1, "r"),
            (l2, 0, 0, q2, "r"),
        )
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

    def test_missing_symbolic_parameter_fails_before_rng_generation(self, monkeypatch):
        l1 = sp.symbols("l1", positive=True)
        robot = Robot((l1, 0, 0, q1, "r"))
        robot.joint_limits = [(-1.0, 1.0)]
        called = {"rng": False}

        original_default_rng = np.random.default_rng

        def tracking_rng(*args, **kwargs):
            called["rng"] = True
            return original_default_rng(*args, **kwargs)

        monkeypatch.setattr(workspace_module.np.random, "default_rng", tracking_rng)

        with pytest.raises(ValueError, match="l1"):
            sample_workspace(robot, samples=3, seed=1)

        assert called["rng"] is False


def test_numerical_fk_failure_aborts_sampling(monkeypatch):
    robot = make_planar_2r()

    monkeypatch.setattr(
        workspace_module,
        "_prepare_position_model",
        lambda robot, parameters=None: lambda *q: [np.nan, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="sample 0"):
        sample_workspace(robot, samples=5, seed=1)


class TestWorkspaceVisualization:
    @staticmethod
    def make_workspace(points):
        points = np.asarray(points, dtype=float)
        configs = np.linspace(-0.5, 0.5, points.shape[0])[:, None]
        return Workspace(
            points,
            configs,
            ((-1.0, 1.0),),
            seed=1,
        )

    def test_auto_selects_xy_when_z_is_constant(self):
        ws = self.make_workspace([
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [-0.5, 1.0, 0.0],
        ])
        fig, ax = plot_workspace(ws, projection="auto")

        assert ax.name != "3d"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert len(ax.collections[0].get_offsets()) == ws.samples
        plt.close(fig)

    @pytest.mark.parametrize(
        "constant_axis, expected_labels",
        [
            (1, ("X", "Z")),
            (0, ("Y", "Z")),
        ],
    )
    def test_auto_axis_aligned_planar_detection(self, constant_axis, expected_labels):
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 2.0, 1.0],
        ])
        points[:, constant_axis] = 0.25
        ws = self.make_workspace(points)

        fig, ax = plot_workspace(ws, projection="auto")

        assert (ax.get_xlabel(), ax.get_ylabel()) == expected_labels
        plt.close(fig)

    def test_spatial_auto_uses_3d(self):
        ws = self.make_workspace([
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25],
            [-0.5, 1.0, 1.0],
        ])
        fig, ax = plot_workspace(ws, projection="auto")

        assert ax.name == "3d"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_zlabel() == "Z"
        plt.close(fig)

    @pytest.mark.parametrize(
        "projection, labels",
        [
            ("xy", ("X", "Y")),
            ("xz", ("X", "Z")),
            ("yz", ("Y", "Z")),
        ],
    )
    def test_explicit_2d_projections(self, projection, labels):
        ws = self.make_workspace([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ])
        fig, ax = plot_workspace(ws, projection=projection)

        assert (ax.get_xlabel(), ax.get_ylabel()) == labels
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_existing_compatible_axes_are_reused(self):
        ws = self.make_workspace([[0, 0, 0], [1, 1, 0]])
        fig, ax = plt.subplots()

        returned_fig, returned_ax = plot_workspace(ws, projection="xy", ax=ax)

        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_incompatible_axis_rejected(self):
        ws = self.make_workspace([[0, 0, 0], [1, 1, 1]])
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="3D axis"):
            plot_workspace(ws, projection="3d", ax=ax)
        plt.close(fig)

    @pytest.mark.parametrize("projection", ["xyz", "plane", 3])
    def test_invalid_projection_rejected(self, projection):
        ws = self.make_workspace([[0, 0, 0], [1, 1, 0]])
        with pytest.raises(ValueError, match="projection"):
            plot_workspace(ws, projection=projection)
