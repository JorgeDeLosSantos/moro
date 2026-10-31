"""V5-B1 regression tests for classic-DH kinematic joint metadata.

No renderer consumes these data in V5-B1. Tests focus on motion-axis
semantics, finite vectors, defensive copying, and legacy compatibility.
"""

import numpy as np
import pytest
import sympy as sp

from moro.core import Robot
from moro.visualization import (
    DHSegmentData, FrameData, JointData, SceneData, evaluate_robot,
)
from moro.visualization.evaluation import _build_joint_data
from moro.visualization.threejs_backend import _scene_to_payload


def test_joint_data_validates_and_normalizes_direction():
    source_origin = np.array([2.0, -3.0, 4.0])
    source_axis = np.array([0.0, 0.0, -7.0])
    joint = JointData(2, "p", source_origin, source_axis)
    assert joint.index == 2
    assert joint.kind == "p"
    assert joint.origin.shape == (3,)
    assert joint.axis.shape == (3,)
    assert joint.origin.dtype.kind == "f"
    assert joint.axis.dtype.kind == "f"
    np.testing.assert_array_equal(joint.origin, [2.0, -3.0, 4.0])
    np.testing.assert_array_equal(joint.axis, [0.0, 0.0, -1.0])
    assert np.linalg.norm(joint.axis) == pytest.approx(1.0)

    source_origin[:] = 20
    source_axis[:] = 20
    np.testing.assert_array_equal(joint.origin, [2.0, -3.0, 4.0])
    np.testing.assert_array_equal(joint.axis, [0.0, 0.0, -1.0])
    assert not np.shares_memory(source_origin, joint.origin)
    assert not np.shares_memory(source_axis, joint.axis)


@pytest.mark.parametrize("index", [0, -1, 1.5, 1.0, "1", True, None])
def test_joint_data_invalid_index(index):
    with pytest.raises(ValueError, match="positive integer"):
        JointData(index, "r", (0, 0, 0), (0, 0, 1))


@pytest.mark.parametrize("kind", ["R", "P", "revolute", "x", "", None])
def test_joint_data_invalid_kind(kind):
    with pytest.raises(ValueError, match="kind"):
        JointData(1, kind, (0, 0, 0), (0, 0, 1))


@pytest.mark.parametrize("bad", [
    (0, 0), (0, 0, 0, 0),
    (0, 0, float("nan")),
    (float("inf"), 0, 0),
    (1j, 0, 0),
])
def test_joint_data_invalid_origin_shape_or_values(bad):
    with pytest.raises(ValueError, match="origin"):
        JointData(1, "p", bad, (0, 0, 1))


@pytest.mark.parametrize("bad", [
    (0, 0), (0, 0, 0, 0),
    (0, 0, 0),
    (float("nan"), 0, 1),
    (float("inf"), 0, 1),
    (1j, 0, 0),
])
def test_joint_data_invalid_axis_shape_direction_or_values(bad):
    with pytest.raises(ValueError, match="axis"):
        JointData(1, "p", (0, 0, 0), bad)


def test_joint_data_accepts_numpy_integer_and_tiny_finite_axis():
    joint = JointData(np.int64(1), "r", np.zeros(3), (0, 0, 1e-300))
    assert joint.index == 1
    np.testing.assert_array_equal(joint.axis, (0, 0, 1))


def test_planar_rr_joint_axes_are_previous_frame_z_directions():
    q1, q2 = sp.symbols("q1 q2")
    robot = Robot((1.0, 0, 0, q1), (0.7, 0, 0, q2))
    scene = evaluate_robot(robot, {q1: 0.4, q2: -0.9})
    assert len(scene.joint_data) == 2
    assert [(j.index, j.kind) for j in scene.joint_data] == [
        (1, "r"), (2, "r"),
    ]
    for i, joint in enumerate(scene.joint_data):
        np.testing.assert_allclose(joint.origin, scene.frames[i].position)
        np.testing.assert_allclose(joint.axis, scene.frames[i].z)
        np.testing.assert_allclose(joint.axis, [0, 0, 1])
        assert np.isfinite(joint.origin).all()
    # The joint-1 motion axis passes through p0, not p1.
    assert not np.allclose(scene.joint_data[0].origin, scene.frames[1].position)
    assert len(scene.dh_segments) == 4


def test_spatial_rrp_axes_follow_previous_frame_with_nonzero_a_and_twists():
    q1, q2, q3 = sp.symbols("q1 q2 q3")
    robot = Robot(
        (0.4, sp.pi / 2, -0.3, q1, "r"),
        (0.7, -sp.pi / 3, 0.2, q2, "r"),
        (-0.2, sp.pi / 4, q3, 0, "p"),
    )
    scene = evaluate_robot(robot, {q1: 0.5, q2: -0.3, q3: -0.15})
    assert [joint.kind for joint in scene.joint_data] == ["r", "r", "p"]
    assert [joint.index for joint in scene.joint_data] == [1, 2, 3]
    for i, joint in enumerate(scene.joint_data):
        np.testing.assert_allclose(joint.origin, scene.frames[i].position)
        np.testing.assert_allclose(joint.axis, scene.frames[i].z)
        assert np.linalg.norm(joint.axis) == pytest.approx(1)
        assert not np.shares_memory(joint.origin, scene.frames[i].T)
        assert not np.shares_memory(joint.axis, scene.frames[i].T)
    assert not np.allclose(scene.joint_data[1].axis, scene.frames[2].z)
    assert not np.allclose(scene.joint_data[2].origin, scene.frames[3].position)


def test_prismatic_axis_does_not_depend_on_prismatic_extension():
    q1, d2 = sp.symbols("q1 d2")
    robot = Robot((0.5, sp.pi / 2, 0.1, q1, "r"),
                  (0.3, 0, d2, 0, "p"))
    left = evaluate_robot(robot, {q1: 0.7, d2: -0.2})
    right = evaluate_robot(robot, {q1: 0.7, d2: 0.6})
    for name in ("origin", "axis"):
        np.testing.assert_allclose(
            getattr(left.joint_data[1], name),
            getattr(right.joint_data[1], name),
        )
    np.testing.assert_allclose(
        left.joint_data[1].origin, left.dh_segments[2].start
    )
    np.testing.assert_allclose(
        left.dh_segments[2].end - left.dh_segments[2].start,
        -0.2 * left.joint_data[1].axis,
    )
    np.testing.assert_allclose(
        right.dh_segments[2].end - right.dh_segments[2].start,
        0.6 * right.joint_data[1].axis,
    )


def test_axis_metadata_does_not_alias_scene_frames():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, sp.pi / 4, 0.2, q)), {q: 0.1})
    origin = scene.joint_data[0].origin.copy()
    axis = scene.joint_data[0].axis.copy()
    scene.frames[0].T[0, 3] += 10
    scene.frames[0].T[2, 2] = 0.5
    np.testing.assert_array_equal(scene.joint_data[0].origin, origin)
    np.testing.assert_array_equal(scene.joint_data[0].axis, axis)


def test_build_joint_data_rejects_incomplete_frame_sequence():
    q = sp.Symbol("q")
    robot = Robot((1, 0, 0, q))
    with pytest.raises(ValueError, match="one base frame"):
        _build_joint_data(robot, [FrameData(np.eye(4))])


def test_legacy_scene_constructors_and_defaults_remain_compatible():
    zero = np.zeros(3)
    T = np.eye(4)
    seg = DHSegmentData(1, "a", zero.copy(), zero.copy())
    scene = SceneData([zero], [FrameData(T)], 1.0, [seg])
    assert scene.dh_segments == [seg]
    assert scene.joint_data == []
    assert set(_scene_to_payload(scene)) == {
        "joints", "frames", "dimension", "dh_segments"
    }
    first = SceneData([], [], 1)
    second = SceneData([], [], 1)
    first.joint_data.append(JointData(1, "r", (0, 0, 0), (0, 0, 1)))
    assert second.joint_data == []


def test_evaluated_scene_keeps_renderer_payload_shape_in_b1():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((0.5, 0, 0.1, q)), {q: 0})
    assert len(scene.joint_data) == 1
    payload = _scene_to_payload(scene)
    assert set(payload) == {"joints", "frames", "dimension", "dh_segments"}
    assert "joint_data" not in payload
