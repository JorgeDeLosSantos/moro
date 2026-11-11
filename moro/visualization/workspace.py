"""Visualization of sampled Cartesian workspaces and sections."""

import numpy as np
import matplotlib.pyplot as plt

from moro.workspace import Workspace, WorkspaceSection


__all__ = ["plot_workspace", "plot_workspace_section"]


_PROJECTIONS = {"auto", "xy", "xz", "yz", "3d"}


def _normalize_plot_options(marker_size, alpha, show_grid):
    """Validate common plot styling independently of the selected backend."""
    try:
        marker_size = float(marker_size)
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("marker_size and alpha must be real numeric values.") from exc
    if not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be a finite value greater than 0.")
    if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha <= 1.")
    if not isinstance(show_grid, (bool, np.bool_)):
        raise ValueError("show_grid must be a Boolean value.")
    return marker_size, alpha, bool(show_grid)


def _normalize_projection(projection):
    if not isinstance(projection, str):
        raise ValueError("projection must be one of 'auto', 'xy', 'xz', 'yz', or '3d'.")
    projection = projection.lower()
    if projection not in _PROJECTIONS:
        raise ValueError("projection must be one of 'auto', 'xy', 'xz', 'yz', or '3d'.")
    return projection


def _auto_projection(workspace):
    spans = np.array(
        [upper - lower for lower, upper in workspace.bounds],
        dtype=float,
    )
    scale = max(float(np.max(np.abs(spans))), 1.0)
    constant = np.isclose(spans, 0.0, rtol=1e-9, atol=1e-12 * scale)

    if constant[2]:
        return "xy"
    if constant[1]:
        return "xz"
    if constant[0]:
        return "yz"
    return "3d"


def _is_3d_axis(ax):
    return getattr(ax, "name", None) == "3d" or hasattr(ax, "zaxis")


def _prepare_axis(projection, ax, figsize):
    if ax is None:
        if projection == "3d":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    is_3d = _is_3d_axis(ax)
    if projection == "3d" and not is_3d:
        raise ValueError("projection='3d' requires a Matplotlib 3D axis.")
    if projection != "3d" and is_3d:
        raise ValueError("A 2D workspace projection requires a Matplotlib 2D axis.")
    return ax.figure, ax


def _set_equal_3d_limits(ax, bounds):
    centers = np.array(
        [(lower + upper) / 2.0 for lower, upper in bounds],
        dtype=float,
    )
    spans = np.array(
        [upper - lower for lower, upper in bounds],
        dtype=float,
    )
    half_extent = float(np.max(spans)) / 2.0
    if half_extent <= 0.0:
        half_extent = 0.5

    ax.set_xlim(centers[0] - half_extent, centers[0] + half_extent)
    ax.set_ylim(centers[1] - half_extent, centers[1] + half_extent)
    ax.set_zlim(centers[2] - half_extent, centers[2] + half_extent)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_workspace(
    workspace,
    *,
    projection="auto",
    backend="threejs",
    ax=None,
    figsize=None,
    marker_size=8,
    alpha=0.5,
    width=None,
    height=None,
    show_grid=False,
):
    """Render sampled positions (Three.js by default, or Matplotlib)."""
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be an instance of Workspace.")

    if backend not in ("threejs", "matplotlib"):
        raise ValueError("backend must be 'threejs' or 'matplotlib'.")
    projection = _normalize_projection(projection)
    if projection == "auto":
        projection = _auto_projection(workspace)

    marker_size, alpha, show_grid = _normalize_plot_options(
        marker_size, alpha, show_grid
    )

    if backend == "threejs":
        if ax is not None or figsize is not None:
            raise ValueError(
                "ax and figsize are Matplotlib-only; use backend='matplotlib'."
            )
        from .threejs_backend import render_workspace_points
        return render_workspace_points(
            workspace.points,
            projection=projection,
            marker_size=marker_size,
            alpha=alpha,
            width=800 if width is None else width,
            height=600 if height is None else height,
            show_grid=show_grid,
        )

    if width is not None or height is not None:
        raise ValueError(
            "width and height are Three.js-only; use backend='threejs'."
        )
    fig, ax = _prepare_axis(projection, ax, (8, 6) if figsize is None else figsize)
    points = workspace.points

    if projection == "3d":
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=marker_size,
            alpha=alpha,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_equal_3d_limits(ax, workspace.bounds)
        if show_grid:
            ax.grid(True)
        return fig, ax

    axis_map = {
        "xy": (0, 1, "X", "Y"),
        "xz": (0, 2, "X", "Z"),
        "yz": (1, 2, "Y", "Z"),
    }
    i, j, xlabel, ylabel = axis_map[projection]
    ax.scatter(points[:, i], points[:, j], s=marker_size, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    if show_grid:
        ax.grid(True)
    return fig, ax


def plot_workspace_section(
    section,
    *,
    backend="threejs",
    ax=None,
    figsize=None,
    marker_size=8,
    alpha=0.5,
    width=None,
    height=None,
    show_grid=False,
):
    """Display a Cartesian slab section, using Three.js by default.

    Matplotlib remains available explicitly and returns a figure and axes.
    The point selection and original 3D coordinates are not modified.
    """
    if not isinstance(section, WorkspaceSection):
        raise TypeError("section must be an instance of WorkspaceSection.")
    if backend not in ("threejs", "matplotlib"):
        raise ValueError("backend must be 'threejs' or 'matplotlib'.")
    marker_size, alpha, show_grid = _normalize_plot_options(
        marker_size, alpha, show_grid
    )

    if backend == "threejs":
        if ax is not None or figsize is not None:
            raise ValueError(
                "ax and figsize are Matplotlib-only; use backend='matplotlib'."
            )
        from .threejs_backend import render_workspace_points

        return render_workspace_points(
            section.points,
            projection=section.plane,
            marker_size=marker_size,
            alpha=alpha,
            width=800 if width is None else width,
            height=600 if height is None else height,
            show_grid=show_grid,
            plane_level=section.level,
            section_info={
                "level": section.level,
                "thickness": section.thickness,
                "source_samples": section.source_samples,
            },
        )

    if width is not None or height is not None:
        raise ValueError(
            "width and height are Three.js-only; use backend='threejs'."
        )
    fig, ax = _prepare_axis(
        section.plane, ax, (8, 6) if figsize is None else figsize
    )
    axis_map = {
        "xy": (0, 1, "X", "Y"),
        "xz": (0, 2, "X", "Z"),
        "yz": (1, 2, "Y", "Z"),
    }
    i, j, xlabel, ylabel = axis_map[section.plane]
    points = section.points
    ax.scatter(points[:, i], points[:, j], s=marker_size, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    if show_grid:
        ax.grid(True)
    return fig, ax
