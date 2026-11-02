"""Tests for backend-independent V3-A visualization layout resolution."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    FrameData, SceneData, VisualizationLayout, evaluate_robot, resolve_layout,
)


def legacy_scene(dimension):
    return SceneData([], [FrameData(np.eye(4))], dimension)


def test_layout_ratios_for_scaled_planar_robot():
    q = symbols("q")
    for scale in (1e-6, 0.001, 1.0, 1000.0):
        robot = Robot((0.7 * scale, 0, 0, q))
        layout = resolve_layout([evaluate_robot(robot, {q: 0.3})])
        assert layout.reference_length == pytest.approx(0.7 * scale)
        assert layout.frame_length == pytest.approx(layout.reference_length / 6)
        assert layout.joint_radius == pytest.approx(0.02 * layout.reference_length)
        assert layout.base_radius == pytest.approx(0.03 * layout.reference_length)
        assert layout.link_radius == pytest.approx(0.015 * layout.reference_length)


def test_global_prismatic_animation_scale_is_fixed():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: d}) for d in (0, 0.1, 0.4, 0.2)]
    layout = resolve_layout(scenes)
    assert layout.reference_length == pytest.approx(0.9)
    assert layout.joint_radius == pytest.approx(0.018)
    assert layout == resolve_layout(scenes)


def test_folding_does_not_change_nominal_sizes():
    q1, q2 = symbols("q1 q2")
    robot = Robot((1.0, 0, 0, q1), (1.0, 0, 0, q2))
    straight = resolve_layout([evaluate_robot(robot, {q1: 0, q2: 0})])
    folded = resolve_layout([evaluate_robot(robot, {q1: 0, q2: np.pi})])
    assert straight == folded


def test_explicit_scale_takes_priority():
    scene = legacy_scene(500)
    layout = resolve_layout([scene], reference_length=2.5)
    assert layout.reference_length == 2.5
    assert layout.frame_length == pytest.approx(2.5 / 6)


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"),
                                        -float("inf"), "bad", True])
def test_invalid_explicit_scale_rejected(invalid):
    with pytest.raises(ValueError, match="reference_length"):
        resolve_layout([legacy_scene(2)], reference_length=invalid)


def test_legacy_dimension_and_degenerate_fallback():
    q = symbols("q")
    zero = evaluate_robot(Robot((0, 0, 0, q)), {q: 0})
    assert zero.characteristic_length == 0
    assert resolve_layout([zero]).reference_length == pytest.approx(1.0)

    old = legacy_scene(7.0)
    assert old.characteristic_length is None
    assert resolve_layout([old]).reference_length == 7
    assert resolve_layout([zero, old]).reference_length == 7


@pytest.mark.parametrize("dimension", [0, -2, float("nan"), float("inf")])
def test_visual_fallback_for_unusable_legacy_dimension(dimension):
    layout = resolve_layout([legacy_scene(dimension)])
    assert layout.reference_length == pytest.approx(1.0)
    assert layout.joint_radius == pytest.approx(0.02)


def test_largest_positive_legacy_dimension_selected():
    scenes = [legacy_scene(-3), legacy_scene(4), legacy_scene(2)]
    assert resolve_layout(scenes).reference_length == 4


def test_dh_length_has_precedence_over_legacy_dimensions():
    q = symbols("q")
    robot = Robot((0.3, 0, 0, q))
    measured = evaluate_robot(robot, {q: 0})
    assert resolve_layout([legacy_scene(100), measured]).reference_length == pytest.approx(0.3)


def test_empty_scenes_rejected():
    with pytest.raises(ValueError, match="at least one"):
        resolve_layout([])


def test_layout_is_frozen():
    layout = resolve_layout([legacy_scene(2)])
    assert isinstance(layout, VisualizationLayout)
    with pytest.raises(FrozenInstanceError):
        layout.joint_radius = 3


def test_no_backend_or_scene_mutation():
    scene = legacy_scene(2)
    original_dim = scene.dimension
    _ = resolve_layout([scene])
    assert scene.dimension == original_dim
    assert scene.dh_segments == []


def test_explicit_reference_length_must_be_numeric():
    with pytest.raises(ValueError, match="reference_length"):
        resolve_layout([legacy_scene(2)], reference_length=object())
