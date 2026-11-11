"""API-003/D5 Three.js workspace viewer and Matplotlib migration contract."""

import json
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.display import HTML

from moro.visualization import plot_workspace, plot_workspace_section
from moro.visualization.threejs_backend import _workspace_point_payload
from moro.workspace import Workspace, section_workspace


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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_generated_workspace_javascript_has_valid_syntax():
    html = plot_workspace(cloud()).data
    scripts = re.findall(r"<script>\s*(.*?)\s*</script>", html, flags=re.S)
    assert len(scripts) == 2
    result = subprocess.run(
        ["node", "--check"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def _payload(html):
    match = re.search(r"var data = (\\{.*?\\});\\s*var framing", html, re.S)
    assert match
    return json.loads(match.group(1))


@pytest.mark.parametrize("plane,level", [
    ("xy", 0.3), ("xz", 2.0), ("yz", 1.0),
])
def test_section_threejs_default_projects_to_nominal_plane(plane, level):
    source = Workspace(
        [[0.0, 1.0, 0.2], [1.0, 2.0, 0.3], [2.0, 3.0, 0.4]],
        [[0.0], [0.4], [0.8]], ((-1.0, 1.0),),
    )
    section = section_workspace(
        source, plane=plane, level=level, thickness=0.4,
    )
    original = section.points.copy()
    result = plot_workspace_section(section)
    assert isinstance(result, HTML)
    payload = _payload(result.data)
    assert payload["projection"] == plane
    assert payload["section"] == {
        "level": level, "thickness": 0.4,
        "source_samples": source.samples,
    }
    assert payload["show_grid"] is False
    assert len(payload["positions"]) == section.samples
    normal = {"xy": 2, "xz": 1, "yz": 0}[plane]
    shown = np.asarray(payload["positions"])
    assert np.all(shown[:, normal] == level)
    for i in range(3):
        if i != normal:
            np.testing.assert_allclose(shown[:, i], original[:, i])
    np.testing.assert_array_equal(section.points, original)


def test_empty_section_returns_html_and_explicit_empty_message():
    source = cloud()
    section = section_workspace(
        source, plane="xy", level=10.0, thickness=0.0,
    )
    html = plot_workspace_section(section).data
    p = _payload(html)
    assert p["positions"] == []
    assert p["framing"]["center"] == [0.0, 0.0, 10.0]
    assert p["framing"]["fit_radius"] > 0
    assert "No points in section" in html
    assert p["section"]["source_samples"] == source.samples


def test_optional_grid_switch_and_plane_aware_js():
    a = _payload(plot_workspace(cloud()).data)
    b = _payload(plot_workspace(cloud(), show_grid=True).data)
    section = section_workspace(cloud(), plane="yz", level=1.0, thickness=2.0)
    c = _payload(plot_workspace_section(section, show_grid=True).data)
    assert a["show_grid"] is False
    assert b["show_grid"] is True
    assert c["show_grid"] is True
    page = plot_workspace_section(section, show_grid=True).data
    assert "gridPlane === 'xy'" in page
    assert "gridPlane === 'yz'" in page
    assert "MoroThreeJS.createGrid" in page


def test_section_matplotlib_explicit_and_grid():
    section = section_workspace(cloud(), plane="xy", level=0.0, thickness=0.0)
    fig, ax = plot_workspace_section(
        section, backend="matplotlib", show_grid=True
    )
    try:
        assert ax.figure is fig
        assert any(line.get_visible() for line in ax.get_xgridlines())
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kwargs,pattern", [
    ({"backend": "unknown"}, "backend"),
    ({"ax": "bad"}, "Matplotlib-only"),
    ({"figsize": (4, 4)}, "Matplotlib-only"),
    ({"backend": "matplotlib", "width": 600}, "Three.js-only"),
    ({"backend": "matplotlib", "height": 400}, "Three.js-only"),
    ({"show_grid": 1}, "show_grid"),
    ({"show_grid": "yes"}, "show_grid"),
    ({"width": 0}, "width"),
    ({"height": True}, "height"),
])
def test_section_invalid_backend_or_option(kwargs, pattern):
    section = section_workspace(cloud(), plane="xy", level=0.0, thickness=0)
    with pytest.raises(ValueError, match=re.escape(pattern)):
        plot_workspace_section(section, **kwargs)


def test_plot_workspace_invalid_grid_input():
    with pytest.raises(ValueError, match="show_grid"):
        plot_workspace(cloud(), show_grid=None)


def test_legacy_matplotlib_section_style_without_backend_is_breaking():
    section = section_workspace(cloud(), plane="xy", level=0, thickness=0)
    assert isinstance(plot_workspace_section(section), HTML)


@pytest.mark.parametrize("enabled", [False, True])
def test_matplotlib_workspace_grid_opt_in(enabled):
    fig, ax = plot_workspace(
        cloud(), backend="matplotlib", projection="xy", show_grid=enabled
    )
    try:
        assert any(line.get_visible() for line in ax.get_xgridlines()) is enabled
    finally:
        plt.close(fig)
