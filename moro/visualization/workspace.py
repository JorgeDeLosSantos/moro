"""Matplotlib visualization for sampled workspaces."""

import numpy as np
import matplotlib.pyplot as plt

from moro.workspace import Workspace


__all__ = ["plot_workspace"]


_PROJECTIONS = {"auto", "xy", "xz", "yz", "3d"}


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
    ax=None,
    figsize=(8, 6),
    marker_size=8,
    alpha=0.5,
):
    """Plot a sampled workspace as a Matplotlib point cloud."""
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be an instance of Workspace.")

    projection = _normalize_projection(projection)
    if projection == "auto":
        projection = _auto_projection(workspace)

    try:
        marker_size = float(marker_size)
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("marker_size and alpha must be real numeric values.") from exc
    if not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be a finite value greater than 0.")
    if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha <= 1.")

    fig, ax = _prepare_axis(projection, ax, figsize)
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
    return fig, ax
