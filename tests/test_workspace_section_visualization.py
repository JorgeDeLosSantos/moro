"""Matplotlib visualization of sampled Cartesian workspace sections (D4)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from moro.visualization import plot_workspace_section
from moro.workspace import Workspace, section_workspace


def source():
    points = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.1],
        [2.0, 3.0, 0.2],
        [3.0, 4.0, 0.1],
        [4.0, 5.0, 0.3],
    ])
    configurations = np.arange(5, dtype=float).reshape(-1, 1)
    return Workspace(
        points, configurations, ((-1.0, 5.0),),
    )


@pytest.mark.parametrize(
    "plane,level,thickness,expected_indices,coords,labels",
    [
        ("xy", 0.1, 0.0, [1, 3], (0, 1), ("X", "Y")),
        ("xz", 3.0, 0.0, [2], (0, 2), ("X", "Z")),
        ("yz", 2.0, 0.0, [2], (1, 2), ("Y", "Z")),
    ],
)
def test_section_plot_preserves_coordinates_and_labeling(
    plane, level, thickness, expected_indices, coords, labels
):
    ws = source()
    section = section_workspace(
        ws, plane=plane, level=level, thickness=thickness,
    )
    fig, ax = plot_workspace_section(section, backend="matplotlib", marker_size=11, alpha=0.25)
    try:
        np.testing.assert_array_equal(section.indices, expected_indices)
        plotted = np.asarray(ax.collections[0].get_offsets())
        np.testing.assert_allclose(
            plotted, ws.points[expected_indices][:, list(coords)],
        )
        assert (ax.get_xlabel(), ax.get_ylabel()) == labels
        assert ax.get_aspect() == 1.0
        assert ax.name != "3d"
        assert ax.collections[0].get_alpha() == 0.25
        assert np.all(ax.collections[0].get_sizes() == 11)
    finally:
        plt.close(fig)


def test_empty_section_draws_empty_2d_axes_without_error():
    section = section_workspace(
        source(), plane="yz", level=999.0, thickness=0,
    )
    assert section.samples == 0
    fig, ax = plot_workspace_section(section, backend="matplotlib")
    try:
        assert np.asarray(ax.collections[0].get_offsets()).shape == (0, 2)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Y", "Z")
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_external_axes_and_3d_axis_rejection():
    section = section_workspace(source(), plane="xy", level=0.1, thickness=0)
    fig, ax = plt.subplots()
    try:
        returned_fig, returned_ax = plot_workspace_section(section, backend="matplotlib", ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
    finally:
        plt.close(fig)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    try:
        with pytest.raises(ValueError, match="2D"):
            plot_workspace_section(section, backend="matplotlib", ax=ax3d)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("marker_size,alpha", [
    (0, 0.5), (-1, 0.5), (np.inf, 0.5), (np.nan, 0.5),
    (2, -0.1), (2, 1.1), (2, np.nan), (2, np.inf),
    ("bad", 0.5), (2, "bad"),
])
def test_invalid_visual_options(marker_size, alpha):
    section = section_workspace(source(), plane="xy", level=0, thickness=0)
    with pytest.raises(ValueError):
        plot_workspace_section(section, backend="matplotlib", marker_size=marker_size, alpha=alpha)


def test_wrong_input_type_and_no_data_mutation():
    with pytest.raises(TypeError, match="WorkspaceSection"):
        plot_workspace_section(source(), backend="matplotlib")
    section = section_workspace(source(), plane="xy", level=0.1, thickness=0)
    points = section.points.copy()
    configurations = section.configurations.copy()
    indices = section.indices.copy()
    fig, ax = plot_workspace_section(section, backend="matplotlib")
    plt.close(fig)
    np.testing.assert_array_equal(section.points, points)
    np.testing.assert_array_equal(section.configurations, configurations)
    np.testing.assert_array_equal(section.indices, indices)
