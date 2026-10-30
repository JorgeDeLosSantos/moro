"""Regression tests for evaluated classic-DH translation geometry (Moro 0.6.0 V1)."""

import numpy as np
import pytest
import sympy as sp

from moro.core import Robot
from moro.visualization import DHSegmentData, FrameData, SceneData, evaluate_robot
from moro.visualization.threejs_backend import _scene_to_payload


def _assert_geometry(robot, values):
    scene = evaluate_robot(robot, values)
    assert len(scene.dh_segments) == 2 * robot.dof

    for i, (a, _, d, _) in enumerate(robot.dh_parameters, start=1):
        d_seg, a_seg = scene.dh_segments[2 * (i - 1):2 * i]
        assert isinstance(d_seg, DHSegmentData)
        assert [(s.row_index, s.kind) for s in (d_seg, a_seg)] == [
            (i, "d"), (i, "a")
        ]
        a_value = float(sp.sympify(a).subs(values))
        d_value = float(sp.sympify(d).subs(values))
        prev, current = scene.frames[i - 1], scene.frames[i]

        np.testing.assert_allclose(d_seg.start, prev.position, atol=1e-11)
        np.testing.assert_allclose(d_seg.end - d_seg.start,
                                   d_value * prev.z, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(a_seg.end - a_seg.start,
                                   a_value * current.x, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(d_seg.end, a_seg.start, atol=1e-11)
        np.testing.assert_allclose(a_seg.end, current.position, atol=1e-11)
        np.testing.assert_allclose(np.linalg.norm(d_seg.end - d_seg.start),
                                   abs(d_value), rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(np.linalg.norm(a_seg.end - a_seg.start),
                                   abs(a_value), rtol=1e-10, atol=1e-11)

        for segment in (d_seg, a_seg):
            for endpoint in (segment.start, segment.end):
                assert isinstance(endpoint, np.ndarray)
                assert endpoint.shape == (3,)
                assert endpoint.dtype.kind == "f"
                assert np.isfinite(endpoint).all()
                assert not np.shares_memory(endpoint, prev.T)
                assert not np.shares_memory(endpoint, current.T)
            assert not np.shares_memory(segment.start, segment.end)

        if i < robot.dof:
            next_d = scene.dh_segments[2 * i]
            np.testing.assert_allclose(a_seg.end, next_d.start, atol=1e-11)

    return scene


def test_planar_rr_retains_zero_d_segments():
    q1, q2 = sp.symbols("q1 q2")
    robot = Robot((1.0, 0, 0, q1), (0.8, 0, 0, q2))
    scene = _assert_geometry(robot, {q1: 0.4, q2: -0.7})
    for seg in scene.dh_segments[::2]:
        np.testing.assert_array_equal(seg.start, seg.end)


def test_spatial_rrr_with_twists_and_signed_parameters():
    qs = sp.symbols("q1:4")
    robot = Robot(
        (-0.8, sp.pi / 2, 0.6, qs[0]),
        (1.2, -sp.pi / 3, -0.4, qs[1]),
        (0.5, sp.pi / 4, 0.3, qs[2]),
    )
    _assert_geometry(robot, dict(zip(qs, (0.3, -0.8, 1.1))))


def test_rrp_prismatic_d_depends_on_configuration():
    q1, q2, q3 = sp.symbols("q1 q2 q3")
    robot = Robot(
        (0.4, sp.pi / 2, 0.2, q1, "r"),
        (0.5, 0, 0, q2, "r"),
        (0.0, 0, q3, 0, "p"),
    )
    first = _assert_geometry(robot, {q1: 0.1, q2: -0.2, q3: 0.35})
    second = _assert_geometry(robot, {q1: 0.1, q2: -0.2, q3: 0.75})
    np.testing.assert_allclose(
        np.linalg.norm(first.dh_segments[-2].end - first.dh_segments[-2].start),
        0.35,
    )
    np.testing.assert_allclose(
        np.linalg.norm(second.dh_segments[-2].end - second.dh_segments[-2].start),
        0.75,
    )


def test_symbolic_dh_distances_and_numpy_scalar_substitution():
    a, d, q = sp.symbols("a d q")
    robot = Robot((a, sp.pi / 3, d, q))
    _assert_geometry(robot, {a: np.float64(-0.8), d: np.float64(0.25), q: 0.5})


def test_degenerate_and_folded_robot_has_all_segments():
    q1, q2 = sp.symbols("q1 q2")
    robot = Robot((1.0, 0, 0, q1), (1.0, 0, 0, q2))
    scene = _assert_geometry(robot, {q1: 0, q2: sp.pi})
    np.testing.assert_allclose(scene.frames[0].position,
                               scene.frames[-1].position, atol=1e-11)
    degenerate = Robot((0, 0, 0, q1))
    zero = _assert_geometry(degenerate, {q1: 0})
    assert all(np.array_equal(s.start, s.end) for s in zero.dh_segments)


@pytest.mark.parametrize("bad", [sp.I, sp.oo, sp.nan, float("inf"), float("nan")])
def test_invalid_dh_a_reports_row_and_parameter(bad):
    q = sp.Symbol("q")
    robot = Robot((bad, 0, 0, q))
    # Call the dedicated scalar helper to isolate the intended validation
    # from the existing frame evaluator's pre-existing error behavior.
    from moro.visualization.evaluation import _evaluate_dh_translation

    with pytest.raises(ValueError, match=r"DH row 1 parameter 'a'"):
        _evaluate_dh_translation(bad, {q: 0.0}, 1, "a")


@pytest.mark.parametrize("bad", [sp.I, sp.oo, sp.nan, float("inf"), float("nan")])
def test_invalid_dh_d_reports_row_and_parameter(bad):
    from moro.visualization.evaluation import _evaluate_dh_translation

    with pytest.raises(ValueError, match=r"DH row 2 parameter 'd'"):
        _evaluate_dh_translation(bad, {}, 2, "d")


def test_unresolved_dh_scalar_is_contextual():
    from moro.visualization.evaluation import _evaluate_dh_translation

    with pytest.raises(ValueError, match=r"DH row 3 parameter 'a'"):
        _evaluate_dh_translation(sp.Symbol("L"), {}, 3, "a")


def test_scalar_helper_handles_plain_python_numbers():
    from moro.visualization.evaluation import _evaluate_dh_translation

    assert _evaluate_dh_translation(-0.5, {}, 2, "d") == -0.5


def test_legacy_scene_constructor_and_payload_unchanged():
    scene = SceneData([np.zeros(3)], [FrameData(np.eye(4))], 1.5)
    assert scene.dh_segments == []
    payload = _scene_to_payload(scene)
    assert set(payload) == {"joints", "frames", "dimension"}


def test_dh_segment_endpoints_do_not_alias_frames_or_one_another():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, 0, 0.5, q)), {q: 0.0})
    original = scene.frames[1].position.copy()
    scene.dh_segments[1].end[0] += 10
    np.testing.assert_array_equal(scene.frames[1].position, original)
    np.testing.assert_allclose(scene.dh_segments[0].end,
                               scene.dh_segments[1].start)


def test_scene_segments_have_independent_defaults():
    first = SceneData([], [], 1.0)
    second = SceneData([], [], 1.0)
    first.dh_segments.append(DHSegmentData(1, "d", np.zeros(3), np.zeros(3)))
    assert second.dh_segments == []
