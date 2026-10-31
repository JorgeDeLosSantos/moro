"""Moro 0.6.0 V4-A bounds and backend-independent framing tests."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sympy import symbols

from moro.core import Robot
from moro.visualization import (
    FrameData,
    SceneBounds,
    SceneData,
    VisualizationStyle,
    compute_scene_bounds,
    evaluate_robot,
    resolve_framing,
    resolve_layout,
    union_bounds,
)


def _frame(position):
    T = np.eye(4)
    T[:3, 3] = position
    return FrameData(T)


def _scene(position, *, dimension=1.0):
    return SceneData(
        joints=[np.array(position, dtype=float)],
        frames=[_frame(position)],
        dimension=dimension,
    )


def _style(**kwargs):
    return VisualizationStyle(
        show_frames=False, show_joints=False, show_links=False,
        **kwargs,
    )


def test_scene_bounds_basic_contract_and_immutable_tuple_storage():
    minimum = np.array([-1.0, 2.0, 3.0])
    bounds = SceneBounds(minimum, (5.0, 8.0, 7.0))
    minimum[0] = 20.0
    assert bounds.minimum == (-1.0, 2.0, 3.0)
    assert bounds.maximum == (5.0, 8.0, 7.0)
    assert bounds.center == pytest.approx((2, 5, 5))
    assert bounds.size == pytest.approx((6, 6, 4))
    with pytest.raises(FrozenInstanceError):
        bounds.minimum = (0, 0, 0)


@pytest.mark.parametrize("minimum,maximum", [
    ((2, 0, 0), (1, 1, 1)),
    ((0, 0), (1, 1, 1)),
    ((0, 0, float("nan")), (1, 1, 1)),
    ((0, 0, 0), (float("inf"), 1, 1)),
])
def test_invalid_bounds_rejected(minimum, maximum):
    with pytest.raises(ValueError):
        SceneBounds(minimum, maximum)


def test_union_bounds_and_empty_sequence():
    first = SceneBounds((1, 2, 3), (4, 5, 6))
    second = SceneBounds((-7, 0, 4), (-1, 8, 12))
    combined = union_bounds([first, second])
    assert combined.minimum == (-7, 0, 3)
    assert combined.maximum == (4, 8, 12)
    with pytest.raises(ValueError, match="at least one"):
        union_bounds([])


def test_bounds_include_dh_translation_corner_not_just_frame_origins():
    q = symbols("q")
    scene = evaluate_robot(Robot((2.0, 0, 3.0, q)), {q: 0})
    layout = resolve_layout([scene])
    bounds = compute_scene_bounds(
        scene, layout=layout,
        style=VisualizationStyle(show_frames=False, show_joints=False),
    )
    radius = layout.link_radius
    assert bounds.minimum == pytest.approx((-radius, -radius, -radius))
    assert bounds.maximum == pytest.approx((2 + radius, radius, 3 + radius))


def test_bounds_include_negative_dh_translation_extrema():
    q = symbols("q")
    scene = evaluate_robot(Robot((-2, 0, -3, q)), {q: 0})
    layout = resolve_layout([scene])
    bounds = compute_scene_bounds(
        scene, layout=layout,
        style=VisualizationStyle(show_frames=False, show_joints=False),
    )
    radius = layout.link_radius
    assert bounds.minimum == pytest.approx((-2-radius, -radius, -3-radius))
    assert bounds.maximum == pytest.approx((radius, radius, radius))


def test_legacy_joint_links_fallback_when_no_dh_segments():
    scene = SceneData(
        joints=[np.array([2., -1., 3.]), np.array([5., 6., 3.])],
        frames=[],
        dimension=200.0,
    )
    layout = resolve_layout([scene], reference_length=10.0)
    bounds = compute_scene_bounds(
        scene, layout=layout,
        style=VisualizationStyle(show_frames=False, show_joints=False),
    )
    radius = layout.link_radius
    assert bounds.minimum == pytest.approx((2-radius, -1-radius, 3-radius))
    assert bounds.maximum == pytest.approx((5+radius, 6+radius, 3+radius))


def test_visible_frames_include_axis_tips_and_explicit_scale():
    scene = _scene((100, -50, 20), dimension=900)
    bounds = compute_scene_bounds(
        scene,
        layout=resolve_layout([scene], reference_length=10),
        style=VisualizationStyle(
            show_links=False, show_joints=False, frame_scale=2.0
        ),
    )
    assert bounds.minimum == pytest.approx((100, -50, 20))
    assert bounds.maximum == pytest.approx((102, -48, 22))


def test_hidden_base_does_not_contribute_to_bounds():
    scene = SceneData(
        joints=[np.array([0., 0., 0.]), np.array([1., 0., 0.])],
        frames=[], dimension=100,
    )
    bounds = compute_scene_bounds(
        scene, layout=resolve_layout([scene], reference_length=10),
        style=VisualizationStyle(
            show_frames=False, show_links=False, show_base=False
        ),
    )
    assert bounds.minimum == pytest.approx((0.8, -0.2, -0.2))
    assert bounds.maximum == pytest.approx((1.2, 0.2, 0.2))


def test_all_hidden_and_empty_scene_use_origin_without_absolute_floor():
    scene = SceneData([], [], 500.0)
    bounds = compute_scene_bounds(scene, style=_style())
    assert bounds.minimum == (0, 0, 0)
    assert bounds.maximum == (0, 0, 0)
    framing = resolve_framing([scene], style=_style())
    assert framing.half_extent == pytest.approx(55.0)
    assert framing.fit_radius == pytest.approx(55.0)


def test_trajectory_controls_inclusion_when_robot_primitives_hidden():
    a = _scene((1, 2, 3))
    b = _scene((5, -4, 7))
    hidden = _style(show_trajectory=False)
    shown = _style(show_trajectory=True, trajectory_mode="trace")
    assert compute_scene_bounds(b, style=hidden).center == (0, 0, 0)
    assert compute_scene_bounds(b, style=shown).center == (5, -4, 7)
    framing = resolve_framing(
        [a, b],
        layout=resolve_layout([a, b], reference_length=10),
        style=shown,
    )
    assert framing.bounds.minimum == (1, -4, 3)
    assert framing.bounds.maximum == (5, 2, 7)
    assert framing.center == pytest.approx((3, -1, 5))


def test_resolve_framing_is_centered_on_offset_geometry_not_origin():
    scene = SceneData(
        joints=[np.array([100., -50., 20.]), np.array([104., -44., 22.])],
        frames=[], dimension=10000,
    )
    layout = resolve_layout([scene], reference_length=10)
    framing = resolve_framing(
        [scene], layout=layout,
        style=VisualizationStyle(show_frames=False, show_joints=False),
    )
    radius = layout.link_radius
    assert framing.center == pytest.approx((102, -47, 21))
    assert framing.bounds.size == pytest.approx(
        (4 + 2*radius, 6 + 2*radius, 2 + 2*radius)
    )
    half_sizes = [s / 2 for s in framing.bounds.size]
    assert framing.half_extent == pytest.approx(1.1 * max(half_sizes))
    assert framing.fit_radius == pytest.approx(
        1.1 * np.linalg.norm(half_sizes)
    )


def test_one_global_framing_covers_every_prismatic_sample():
    q = symbols("q")
    robot = Robot((0.5, 0, q, 0, "p"))
    scenes = [evaluate_robot(robot, {q: x}) for x in (0., 0.1, 0.4, 0.2)]
    layout = resolve_layout(scenes)
    framing = resolve_framing(
        scenes, layout=layout,
        style=VisualizationStyle(show_frames=False, show_joints=False),
    )
    for scene in scenes:
        individual = compute_scene_bounds(
            scene, layout=layout,
            style=VisualizationStyle(show_frames=False, show_joints=False),
        )
        for i in range(3):
            assert framing.bounds.minimum[i] <= individual.minimum[i]
            assert framing.bounds.maximum[i] >= individual.maximum[i]
    assert framing.center[2] == pytest.approx(0.2)


def test_framing_scales_linearly_between_meters_and_millimeters():
    q = symbols("q")
    framings = []
    for factor in (1, 1000):
        robot = Robot((0.4*factor, 0, 0.2*factor, q))
        scene = evaluate_robot(robot, {q: 0})
        framings.append(resolve_framing([scene]))
    first, second = framings
    assert second.center == pytest.approx(tuple(1000*x for x in first.center))
    assert second.bounds.size == pytest.approx(
        tuple(1000*x for x in first.bounds.size)
    )
    assert second.half_extent == pytest.approx(first.half_extent*1000)
    assert second.fit_radius == pytest.approx(first.fit_radius*1000)


def test_collapsed_geometry_uses_visual_only_relative_minimum():
    scene = _scene((0, 0, 0))
    layout = resolve_layout([scene], reference_length=7)
    framing = resolve_framing([scene], layout=layout, style=_style())
    assert framing.bounds.size == (0, 0, 0)
    assert framing.half_extent == pytest.approx(0.77)
    assert framing.fit_radius == pytest.approx(0.77)


@pytest.mark.parametrize("kwargs", [
    {"margin": -0.01},
    {"margin": float("nan")},
    {"margin": float("inf")},
    {"min_extent_factor": -1},
    {"min_extent_factor": float("nan")},
])
def test_invalid_framing_factors_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_framing([_scene((0, 0, 0))], **kwargs)


def test_empty_framing_sequence_rejected():
    with pytest.raises(ValueError, match="at least one"):
        resolve_framing([])


def test_bounding_rejects_invalid_geometry_values():
    scene = _scene((0, 0, 0))
    scene.frames[0].T[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_scene_bounds(
            scene, style=VisualizationStyle(
                show_links=False, show_joints=False, show_frames=True,
            ),
        )


def test_scene_bounds_do_not_alias_mutable_scene_points():
    scene = _scene((1, 2, 3))
    bounds = compute_scene_bounds(
        scene, style=VisualizationStyle(
            show_links=False, show_frames=False, show_joints=True,
        )
    )
    scene.joints[0][0] = 100
    assert bounds.center[0] == pytest.approx(1)


def test_framing_does_not_change_scene_dimension_or_layout():
    q = symbols("q")
    scene = evaluate_robot(Robot((0.5, 0, 0, q)), {q: 0})
    layout = resolve_layout([scene])
    dim = scene.dimension
    framing = resolve_framing([scene], layout=layout)
    assert scene.dimension == dim
    assert layout.reference_length == pytest.approx(0.5)
    assert framing.bounds.size[0] > 0
