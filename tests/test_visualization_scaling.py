"""Tests for Moro 0.6.0 V2 geometric scale metrics."""

import numpy as np
import pytest
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    DHSegmentData,
    FrameData,
    SceneData,
    evaluate_robot,
    resolve_reference_length,
)


def test_characteristic_length_planar_rr_and_units():
    q1, q2 = symbols("q1 q2")
    meters = Robot((0.4, 0, 0, q1), (0.3, 0, 0, q2, "r"))
    millimeters = Robot((400, 0, 0, q1), (300, 0, 0, q2, "r"))
    folded = {q1: 0, q2: np.pi}
    assert evaluate_robot(meters, folded).characteristic_length == pytest.approx(0.7)
    assert evaluate_robot(millimeters, folded).characteristic_length == pytest.approx(700)


def test_scale_depends_on_segments_not_end_effector_position():
    q1, q2 = symbols("q1 q2")
    robot = Robot((1, 0, 0, q1), (1, 0, 0, q2))
    straight = evaluate_robot(robot, {q1: 0, q2: 0})
    folded = evaluate_robot(robot, {q1: 0, q2: np.pi})
    assert straight.characteristic_length == pytest.approx(2)
    assert folded.characteristic_length == pytest.approx(2)
    np.testing.assert_allclose(folded.joints[-1], [0, 0, 0], atol=1e-12)


def test_prismatic_reference_is_maximum_across_animation():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: d}) for d in (0, 0.1, 0.4, 0.2)]
    assert [s.characteristic_length for s in scenes] == pytest.approx(
        [0.5, 0.6, 0.9, 0.7]
    )
    assert resolve_reference_length(scenes) == pytest.approx(0.9)
    assert resolve_reference_length(iter(scenes)) == pytest.approx(0.9)


def test_zero_and_absent_geometry_are_distinct():
    q = symbols("q")
    zero = evaluate_robot(Robot((0, 0, 0, q)), {q: 0})
    assert zero.characteristic_length == pytest.approx(0)
    assert resolve_reference_length([zero]) == pytest.approx(0)
    legacy = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 1.0)
    assert legacy.characteristic_length is None
    assert resolve_reference_length([legacy]) is None
    assert resolve_reference_length([]) is None
    assert resolve_reference_length([legacy, zero]) == pytest.approx(0)


def test_characteristic_length_is_computed_after_mutation():
    seg = DHSegmentData(1, "a", np.zeros(3), np.array([3.0, 4.0, 0.0]))
    scene = SceneData([], [], 1.0, [seg])
    assert scene.characteristic_length == pytest.approx(5)
    seg.end[0] = 0
    assert scene.characteristic_length == pytest.approx(4)


def test_negative_dh_lengths_count_as_positive():
    q = symbols("q")
    robot = Robot((-0.3, 0, -0.4, q))
    scene = evaluate_robot(robot, {q: 0})
    assert scene.characteristic_length == pytest.approx(0.7)


def test_large_and_small_unit_scales():
    q = symbols("q")
    for scale in (1e-6, 1e-3, 1, 1e3):
        robot = Robot((scale, 0, 0, q))
        scene = evaluate_robot(robot, {q: 0})
        assert scene.characteristic_length == pytest.approx(scale)


def test_nonfinite_segment_length_is_rejected():
    scene = SceneData([], [], 1.0, [
        DHSegmentData(1, "a", np.zeros(3), np.array([np.nan, 0, 0]))
    ])
    with pytest.raises(ValueError, match="finite"):
        _ = scene.characteristic_length
