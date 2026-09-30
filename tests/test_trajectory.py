"""Tests for Moro numerical trajectory generation."""

import numpy as np
import pytest

from moro.trajectory import (
    JointTrajectory,
    PositionTrajectory,
    joint_trajectory,
    position_trajectory,
)


class TestJointTrajectoryResult:
    def test_valid_result_and_properties(self):
        t = np.array([0.0, 0.5, 1.0])
        q = np.array([[0.0], [0.5], [1.0]])
        qd = np.ones((3, 1))
        qdd = np.zeros((3, 1))

        traj = JointTrajectory(t, q, qd, qdd, "linear")

        assert traj.samples == 3
        assert traj.duration == pytest.approx(1.0)
        assert traj.dof == 1
        assert traj.q.shape == (3, 1)
        assert "JointTrajectory" in repr(traj)
        assert "samples=3" in repr(traj)
        assert "dof=1" in repr(traj)

    def test_defensive_copying(self):
        t = np.array([0.0, 1.0])
        q = np.array([[0.0], [1.0]])
        qd = np.array([[1.0], [1.0]])
        qdd = np.zeros((2, 1))

        traj = JointTrajectory(t, q, qd, qdd, "linear")

        t[:] = -1.0
        q[:] = -2.0

        np.testing.assert_allclose(traj.t, [0.0, 1.0])
        np.testing.assert_allclose(traj.q[:, 0], [0.0, 1.0])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="identical shapes"):
            JointTrajectory(
                [0.0, 1.0],
                [[0.0], [1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0], [0.0]],
                "linear",
            )

    def test_rejects_sample_count_mismatch(self):
        with pytest.raises(ValueError, match="one row per time sample"):
            JointTrajectory(
                [0.0, 0.5, 1.0],
                [[0.0], [1.0]],
                [[1.0], [1.0]],
                [[0.0], [0.0]],
                "linear",
            )


class TestPositionTrajectoryResult:
    def test_valid_result_and_properties(self):
        t = [1.0, 2.0, 3.0]
        p = np.zeros((3, 3))
        v = np.zeros((3, 3))
        a = np.zeros((3, 3))

        traj = PositionTrajectory(t, p, v, a, "quintic")

        assert traj.samples == 3
        assert traj.duration == pytest.approx(2.0)
        assert traj.p.shape == (3, 3)
        assert "PositionTrajectory" in repr(traj)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (2, 2),
            (2, 4),
            (3, 2),
        ],
    )
    def test_rejects_wrong_cartesian_shapes(self, bad_shape):
        t = [0.0, 1.0]
        p = np.zeros(bad_shape)
        v = np.zeros((2, 3))
        a = np.zeros((2, 3))

        with pytest.raises(ValueError, match=r"shape \(len\(t\), 3\)"):
            PositionTrajectory(t, p, v, a, "quintic")


class TestTimeValidation:
    @pytest.mark.parametrize(
        "t",
        [
            [],
            [0.0],
            [0.0, 0.0],
            [1.0, 0.5],
            [0.0, np.nan],
            [0.0, np.inf],
        ],
    )
    def test_rejects_invalid_time_vectors(self, t):
        with pytest.raises(ValueError):
            joint_trajectory(0.0, 1.0, t)

    def test_accepts_two_samples(self):
        traj = joint_trajectory(0.0, 1.0, [0.0, 1.0])
        assert traj.samples == 2

    def test_accepts_nonuniform_time(self):
        t = [0.0, 0.1, 0.4, 1.0]
        traj = joint_trajectory(0.0, 1.0, t, method="cubic")
        np.testing.assert_allclose(traj.t, t)

    def test_time_origin_invariance(self):
        t1 = np.linspace(0.0, 2.0, 31)
        t2 = np.linspace(10.0, 12.0, 31)

        a = joint_trajectory(0.0, 1.0, t1, method="quintic")
        b = joint_trajectory(0.0, 1.0, t2, method="quintic")

        np.testing.assert_allclose(a.q, b.q)
        np.testing.assert_allclose(a.qd, b.qd)
        np.testing.assert_allclose(a.qdd, b.qdd)


class TestMethodValidation:
    @pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
    def test_accepts_all_methods(self, method):
        traj = joint_trajectory(0.0, 1.0, [0.0, 1.0], method=method)
        assert traj.method == method

    def test_normalizes_capitalization(self):
        traj = joint_trajectory(0.0, 1.0, [0.0, 1.0], method="QuInTiC")
        assert traj.method == "quintic"

    def test_rejects_invalid_method(self):
        with pytest.raises(ValueError, match="linear.*cubic.*quintic"):
            joint_trajectory(0.0, 1.0, [0.0, 1.0], method="spline")


class TestJointInputValidation:
    def test_scalar_1dof_output_shape(self):
        traj = joint_trajectory(0.0, 1.0, np.linspace(0, 1, 11))
        assert traj.q.shape == (11, 1)
        assert traj.qd.shape == (11, 1)
        assert traj.qdd.shape == (11, 1)

    def test_rejects_mismatched_endpoint_dimensions(self):
        with pytest.raises(ValueError, match="same number"):
            joint_trajectory([0.0, 0.0], [1.0], [0.0, 1.0])

    def test_rejects_empty_joint_vector(self):
        with pytest.raises(ValueError, match="at least one"):
            joint_trajectory([], [], [0.0, 1.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, 1 + 2j])
    def test_rejects_bad_joint_endpoint_values(self, bad):
        with pytest.raises(ValueError):
            joint_trajectory([0.0, bad], [1.0, 2.0], [0.0, 1.0])

    def test_accepts_scalar_derivative_for_1dof(self):
        traj = joint_trajectory(
            0.0,
            1.0,
            [0.0, 1.0],
            method="cubic",
            qd0=0.5,
            qdf=-0.25,
        )
        assert traj.qd[0, 0] == pytest.approx(0.5)
        assert traj.qd[-1, 0] == pytest.approx(-0.25)

    def test_rejects_scalar_broadcasting_for_multidof(self):
        with pytest.raises(ValueError, match="broadcasting"):
            joint_trajectory(
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                method="cubic",
                qd0=0.5,
            )


class TestCartesianInputValidation:
    @pytest.mark.parametrize("p0", [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 0.0])
    def test_rejects_non_three_component_position(self, p0):
        with pytest.raises(ValueError):
            position_trajectory(
                p0,
                [1.0, 1.0, 1.0],
                [0.0, 1.0],
            )

    def test_rejects_malformed_boundary_vector(self):
        with pytest.raises(ValueError, match="exactly 3"):
            position_trajectory(
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0],
                method="cubic",
                v0=[0.0, 0.0],
            )


class TestLinearTrajectory:
    def test_classical_profile_and_derivatives(self):
        t = np.linspace(0.0, 2.0, 21)
        traj = joint_trajectory(0.0, 4.0, t, method="linear")

        tau = (t - t[0]) / (t[-1] - t[0])
        np.testing.assert_allclose(traj.q[:, 0], 4.0 * tau)
        np.testing.assert_allclose(traj.qd[:, 0], 2.0)
        np.testing.assert_allclose(traj.qdd[:, 0], 0.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"qd0": 0.0},
            {"qdf": 0.0},
            {"qdd0": 0.0},
            {"qddf": 0.0},
        ],
    )
    def test_rejects_explicit_derivative_conditions(self, kwargs):
        with pytest.raises(ValueError, match="positions only"):
            joint_trajectory(
                0.0,
                1.0,
                [0.0, 1.0],
                method="linear",
                **kwargs,
            )


class TestCubicTrajectory:
    def test_classical_zero_velocity_profile(self):
        t = np.linspace(0.0, 1.0, 31)
        tau = t[:, None]
        traj = joint_trajectory(0.0, 1.0, t, method="cubic")

        expected = 3 * tau**2 - 2 * tau**3
        np.testing.assert_allclose(traj.q, expected)
        assert traj.qd[0, 0] == pytest.approx(0.0)
        assert traj.qd[-1, 0] == pytest.approx(0.0)

    def test_nonzero_endpoint_velocities(self):
        traj = joint_trajectory(
            [0.0, 1.0],
            [2.0, -1.0],
            np.linspace(2.0, 5.0, 41),
            method="cubic",
            qd0=[0.2, -0.1],
            qdf=[-0.3, 0.4],
        )

        np.testing.assert_allclose(traj.q[0], [0.0, 1.0])
        np.testing.assert_allclose(traj.q[-1], [2.0, -1.0])
        np.testing.assert_allclose(traj.qd[0], [0.2, -0.1], atol=1e-12)
        np.testing.assert_allclose(traj.qd[-1], [-0.3, 0.4], atol=1e-12)

    def test_partial_velocity_defaults(self):
        traj = joint_trajectory(
            0.0,
            1.0,
            [0.0, 1.0],
            method="cubic",
            qd0=0.4,
        )
        assert traj.qd[0, 0] == pytest.approx(0.4)
        assert traj.qd[-1, 0] == pytest.approx(0.0)

    def test_rejects_acceleration_conditions(self):
        with pytest.raises(ValueError, match="does not support acceleration"):
            joint_trajectory(
                0.0,
                1.0,
                [0.0, 1.0],
                method="cubic",
                qdd0=0.0,
            )


class TestQuinticTrajectory:
    def test_classical_zero_derivative_profile(self):
        t = np.linspace(0.0, 1.0, 31)
        tau = t[:, None]
        traj = joint_trajectory(0.0, 1.0, t, method="quintic")

        expected = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        np.testing.assert_allclose(traj.q, expected, atol=1e-12)
        np.testing.assert_allclose(traj.qd[[0, -1]], 0.0, atol=1e-12)
        np.testing.assert_allclose(traj.qdd[[0, -1]], 0.0, atol=1e-12)

    def test_nonzero_velocity_and_acceleration_boundaries(self):
        traj = joint_trajectory(
            [0.0, 1.0],
            [2.0, -1.0],
            np.linspace(0.0, 2.0, 51),
            method="quintic",
            qd0=[0.2, -0.1],
            qdf=[-0.3, 0.4],
            qdd0=[0.5, 0.0],
            qddf=[-0.25, 0.2],
        )

        np.testing.assert_allclose(traj.q[0], [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(traj.q[-1], [2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(traj.qd[0], [0.2, -0.1], atol=1e-12)
        np.testing.assert_allclose(traj.qd[-1], [-0.3, 0.4], atol=1e-12)
        np.testing.assert_allclose(traj.qdd[0], [0.5, 0.0], atol=1e-11)
        np.testing.assert_allclose(traj.qdd[-1], [-0.25, 0.2], atol=1e-11)


class TestTimeScaling:
    @pytest.mark.parametrize("method", ["linear", "cubic", "quintic"])
    def test_duration_scaling(self, method):
        tau = np.linspace(0.0, 1.0, 31)
        t1 = tau
        t2 = 2.0 * tau

        a = joint_trajectory(0.0, 1.0, t1, method=method)
        b = joint_trajectory(0.0, 1.0, t2, method=method)

        np.testing.assert_allclose(a.q, b.q)
        np.testing.assert_allclose(a.qd, 2.0 * b.qd, atol=1e-12)
        np.testing.assert_allclose(a.qdd, 4.0 * b.qdd, atol=1e-11)


class TestPositionTrajectoryMathematics:
    def test_zero_derivative_cartesian_path_is_line_segment(self):
        p0 = np.array([0.0, 1.0, -0.5])
        pf = np.array([2.0, 3.0, 0.5])
        traj = position_trajectory(
            p0,
            pf,
            np.linspace(0.0, 2.0, 41),
            method="quintic",
        )

        delta = pf - p0
        for p in traj.p:
            alpha_candidates = (p - p0) / delta
            np.testing.assert_allclose(
                alpha_candidates,
                np.full(3, alpha_candidates[0]),
                atol=1e-10,
            )

    def test_transverse_boundary_velocity_can_curve_path(self):
        traj = position_trajectory(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            np.linspace(0.0, 1.0, 21),
            method="cubic",
            v0=[0.0, 1.0, 0.0],
            vf=[0.0, 0.0, 0.0],
        )

        assert np.max(np.abs(traj.p[:, 1])) > 1e-6
        np.testing.assert_allclose(traj.p[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(traj.p[-1], [1.0, 0.0, 0.0])


def test_multidof_constant_coordinate_remains_constant():
    traj = joint_trajectory(
        [0.0, 2.0, -1.0],
        [1.0, 2.0, 3.0],
        np.linspace(0.0, 1.0, 31),
        method="quintic",
    )

    np.testing.assert_allclose(traj.q[:, 1], 2.0)
    np.testing.assert_allclose(traj.qd[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj.qdd[:, 1], 0.0, atol=1e-12)
