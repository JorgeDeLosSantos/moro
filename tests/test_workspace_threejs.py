"""API-003/D5 Three.js workspace viewer and Matplotlib migration contract."""

import json
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.display import HTML

from moro.visualization import plot_workspace
from moro.visualization.threejs_backend import _workspace_point_payload
from moro.workspace import Workspace


def cloud():
    pts = [[-2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [2.0, -1.0, 0.0]]
    return Workspace(pts, [[0.0], [0.5], [1.0]], ((-1.0, 2.0),))


@pytest.mark.parametrize("projection,expected", [
    ("auto", "xy"), ("xy", "xy"), ("xz", "xz"),
    ("yz", "yz"), ("3d", "3d"),
])
def test_threejs_default_and_projection_payload(projection, expected):
    result = plot_workspace(cloud(), projection=projection)
    assert isinstance(result, HTML)
    html = result.data
    assert "window.MoroThreeJS.ensureThreeReady" in html
    assert "THREE.BufferGeometry" in html
    assert "THREE.PointsMaterial" in html
    assert "THREE.Float32BufferAttribute" in html
    assert "fitCameraView" in html
    assert "setPresetView" in html
    assert "__COMMON_SCRIPT__" not in html
    assert "__WORKSPACE_DATA__" not in html
    match = re.search(r"var data = (\{.*?\});\s*var framing", html, flags=re.S)
    assert match
    payload = json.loads(match.group(1))
    assert payload["projection"] == expected
    assert len(payload["positions"]) == 3
    assert payload["marker_size"] == 8
    assert payload["alpha"] == 0.5
    assert np.all(np.isfinite(payload["framing"]["center"]))
    assert payload["framing"]["fit_radius"] > 0


@pytest.mark.parametrize("projection,normal", [
    ("xy", 2), ("xz", 1), ("yz", 0),
])
def test_projection_collapses_normal_axis_without_mutating_source(projection, normal):
    positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    before = positions.copy()
    payload = _workspace_point_payload(positions, projection=projection)
    displayed = np.asarray(payload["positions"])
    assert displayed[0, normal] == displayed[1, normal]
    other = [i for i in range(3) if i != normal]
    np.testing.assert_array_equal(displayed[:, other], before[:, other])
    np.testing.assert_array_equal(positions, before)


def test_spatial_cloud_default_keeps_all_coordinates():
    points = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 0.0]])
    ws = Workspace(points, [[0.0], [0.5]], ((-1.0, 1.0),))
    payload = _workspace_point_payload(ws.points, projection="3d")
    np.testing.assert_array_equal(payload["positions"], points)
    assert isinstance(plot_workspace(ws), HTML)


def test_degenerate_one_point_and_tiny_scale_have_valid_camera():
    a = _workspace_point_payload([[0.0, 0.0, 0.0]], projection="xy")
    b = _workspace_point_payload([[1e-9, -2e-9, 3e-9]], projection="3d")
    assert a["framing"]["fit_radius"] > 0
    assert b["framing"]["fit_radius"] > 0


def test_matplotlib_is_explicit_and_preserves_result_contract():
    fig, ax = plot_workspace(cloud(), backend="matplotlib", projection="xy")
    try:
        assert ax.figure is fig
        assert ax.get_aspect() == 1.0
        assert len(ax.collections[0].get_offsets()) == 3
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kwargs,pattern", [
    ({"backend": "bad"}, "backend"),
    ({"backend": "threejs", "ax": "unused"}, "Matplotlib-only"),
    ({"backend": "threejs", "figsize": (5, 5)}, "Matplotlib-only"),
    ({"backend": "matplotlib", "width": 600}, "Three.js-only"),
    ({"backend": "matplotlib", "height": 400}, "Three.js-only"),
    ({"backend": "threejs", "width": True}, "width"),
    ({"backend": "threejs", "width": -5}, "width"),
    ({"backend": "threejs", "height": 0}, "height"),
    ({"backend": "threejs", "height": 4.2}, "height"),
    ({"backend": "threejs", "marker_size": np.inf}, "marker_size"),
    ({"backend": "threejs", "alpha": 1.1}, "alpha"),
])
def test_backend_parameters_fail_instead_of_being_silently_ignored(kwargs, pattern):
    with pytest.raises(ValueError, match=re.escape(pattern)):
        plot_workspace(cloud(), **kwargs)


def test_width_height_and_marker_style_serialization():
    html = plot_workspace(
        cloud(), backend="threejs", width=960, height=540,
        marker_size=5, alpha=0.25,
    ).data
    assert "var width = 960, height = 540" in html
    match = re.search(r"var data = (\{.*?\});\s*var framing", html, re.S)
    assert match
    p = json.loads(match.group(1))
    assert p["marker_size"] == 5
    assert p["alpha"] == 0.25
