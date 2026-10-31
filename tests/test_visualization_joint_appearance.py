"""V5-B2 experimental Three.js cylinder/prism/hybrid joint appearances."""

import json
import re

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import sympy as sp

from moro.core import Robot
from moro.visualization import (
    MatplotlibBackend, ThreeJSBackend, VisualizationStyle,
    compute_scene_bounds, evaluate_robot, resolve_layout,
)
from moro.visualization.joint_solids import (
    build_joint_solids, solid_bounding_radius,
)


def _robot():
    q1, q2, d3 = sp.symbols("q1 q2 d3")
    robot = Robot(
        (0.5, sp.pi / 2, 0.2, q1, "r"),
        (0.4, -sp.pi / 3, 0.1, q2, "r"),
        (0.3, 0, d3, 0, "p"),
    )
    return robot, (q1, q2, d3)


def _style(appearance):
    return VisualizationStyle(
        joint_representation="kinematic",
        joint_appearance=appearance,
        joint_scale=1.1,
        show_frames=False,
        show_links=False,
    )


def _extract(html, variable):
    result = re.search(r"var " + variable + r" = (\{[^\n]+\});", html)
    assert result is not None
    return json.loads(result.group(1))


@pytest.mark.parametrize("appearance", ["schematic", "solid", "hybrid"])
def test_threejs_three_appearances_have_expected_payloads(appearance):
    robot, qs = _robot()
    scene = evaluate_robot(robot, dict(zip(qs, [0.3, -0.5, 0.2])))
    style = _style(appearance)
    payload = _extract(ThreeJSBackend.render(scene, style=style).data, "data")
    assert payload["style"]["joint_appearance"] == appearance
    assert payload["style"]["joint_representation"] == "kinematic"
    assert ("joint_symbols" in payload) == (appearance != "solid")
    assert ("joint_solids" in payload) == (appearance != "schematic")
    if "joint_solids" in payload:
        assert [item["kind"] for item in payload["joint_solids"]] == [
            "r", "r", "p"
        ]


def test_simple_default_mode_is_unchanged():
    robot, qs = _robot()
    scene = evaluate_robot(robot, dict(zip(qs, [0, 0, 0.2])))
    payload = _extract(ThreeJSBackend.render(scene).data, "data")
    assert payload["style"]["joint_representation"] == "simple"
    assert payload["style"]["joint_appearance"] == "schematic"
    assert "joint_solids" not in payload
    assert "joint_symbols" not in payload


@pytest.mark.parametrize("appearance", ["solid", "hybrid"])
def test_threejs_solid_dimensions_and_orientation_metadata(appearance):
    robot, qs = _robot()
    scene = evaluate_robot(robot, dict(zip(qs, [0.4, -0.2, -0.5])))
    layout = resolve_layout([scene])
    solids = build_joint_solids(scene, layout=layout, style=_style(appearance))
    assert len(solids) == 3
    assert [solid["index"] for solid in solids] == [1, 2, 3]
    for solid, joint in zip(solids, scene.joint_data):
        np.testing.assert_allclose(solid["origin"], joint.origin)
        np.testing.assert_allclose(solid["axis"], joint.axis)
        assert np.linalg.norm(solid["transverse"]) == pytest.approx(1.0)
        assert np.dot(solid["transverse"], joint.axis) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            solid["transverse"], scene.frames[joint.index - 1].x, atol=1e-12
        )
        assert solid["length"] > 0
        assert solid_bounding_radius(solid) > 0
        if joint.kind == "r":
            assert solid["radius"] > 0
            assert "width" not in solid
        else:
            assert solid["width"] > 0 and solid["depth"] > 0
            assert "radius" not in solid


@pytest.mark.parametrize("appearance", ["solid", "hybrid"])
def test_global_framing_contains_all_solid_primitive_bounds(appearance):
    robot, qs = _robot()
    scene = evaluate_robot(robot, dict(zip(qs, [-0.2, 0.6, -0.1])))
    layout = resolve_layout([scene])
    style = _style(appearance)
    bounds = compute_scene_bounds(scene, layout=layout, style=style)
    for solid in build_joint_solids(scene, layout=layout, style=style):
        radius = solid_bounding_radius(solid)
        for i, origin in enumerate(solid["origin"]):
            assert bounds.minimum[i] <= origin - radius + 1e-12
            assert bounds.maximum[i] >= origin + radius - 1e-12


@pytest.mark.parametrize("appearance", ["solid", "hybrid"])
def test_animation_preserves_solid_identity_and_dimensions(appearance):
    robot, qs = _robot()
    scenes = [
        evaluate_robot(robot, dict(zip(qs, values)))
        for values in ((0, 0, 0), (0.5, -0.2, 0.4), (-0.2, 0.8, -0.5))
    ]
    payload = _extract(
        ThreeJSBackend.animate(scenes, style=_style(appearance)).data,
        "animationData",
    )
    for frame in payload["frames"]:
        assert [s["kind"] for s in frame["joint_solids"]] == ["r", "r", "p"]
    expected_dims = [
        (s["kind"], s["length"], s.get("radius"), s.get("width"), s.get("depth"))
        for s in payload["frames"][0]["joint_solids"]
    ]
    for frame in payload["frames"][1:]:
        dims = [
            (s["kind"], s["length"], s.get("radius"), s.get("width"), s.get("depth"))
            for s in frame["joint_solids"]
        ]
        assert dims == expected_dims
    if appearance == "solid":
        assert "joint_symbols" not in payload["frames"][0]
    else:
        assert "joint_symbols" in payload["frames"][0]


def test_solid_descriptors_covary_between_meters_and_millimeters():
    q, d = sp.symbols("q d")
    data = []
    for factor in (1, 1000):
        robot = Robot(
            (0.5 * factor, sp.pi / 2, 0.1 * factor, q, "r"),
            (0.2 * factor, 0, d, 0, "p"),
        )
        scene = evaluate_robot(robot, {q: .6, d: .3 * factor})
        data.append(build_joint_solids(
            scene, layout=resolve_layout([scene]), style=_style("solid")
        ))
    for a, b in zip(data[0], data[1]):
        assert a["kind"] == b["kind"]
        np.testing.assert_allclose(b["origin"], 1000 * np.asarray(a["origin"]))
        np.testing.assert_allclose(b["axis"], a["axis"])
        np.testing.assert_allclose(b["transverse"], a["transverse"])
        for key in ("radius", "width", "depth", "length"):
            if key in a:
                assert b[key] == pytest.approx(1000 * a[key])


@pytest.mark.parametrize("appearance", ["solid", "hybrid"])
def test_matplotlib_rejects_experimental_threejs_only_variants(appearance):
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, 0, 0, q)), {q: .1})
    with pytest.raises(ValueError, match="Three.js-only"):
        MatplotlibBackend.render(scene, style=_style(appearance))


@pytest.mark.parametrize("bad", ["mesh", "cylinder", "", None])
def test_invalid_appearance_rejected(bad):
    with pytest.raises(ValueError, match="joint_appearance"):
        VisualizationStyle(joint_appearance=bad)


def test_schematic_matplotlib_still_works_and_solid_can_be_hidden():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, 0, 0, q)), {q: 0})
    _, ax = MatplotlibBackend.render(scene, style=_style("schematic"))
    assert len(ax.lines) > 0
    ax.figure.clf()
    # No symbols means no backend-specific appearance to render.
    style = VisualizationStyle(
        show_joints=False,
        joint_representation="kinematic",
        joint_appearance="solid",
    )
    _, ax = MatplotlibBackend.render(scene, style=style)
    assert len(ax.collections) == 0


def test_solid_geometry_is_inactive_in_simple_representation():
    q = sp.Symbol("q")
    scene = evaluate_robot(Robot((1, 0, 0, q)), {q: 0})
    style = VisualizationStyle(
        joint_representation="simple", joint_appearance="hybrid"
    )
    assert build_joint_solids(scene, layout=resolve_layout([scene]), style=style) == []
