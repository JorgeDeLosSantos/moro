"""Matplotlib visualization backend."""

import numpy as np

from .data import SceneData
from .framing import SceneFraming, resolve_framing
from .layout import VisualizationLayout, resolve_layout
from .style import VisualizationStyle
from .threejs_backend import _extract_end_effector_trajectory


def _marker_area_from_world_radius(
    ax, radius: float, visible_half_extent: float
) -> float:
    """Map a nominal world radius to Matplotlib scatter marker area in pt².

    Matplotlib's scatter markers are display-space objects. This approximate
    conversion uses the *resolved visible span*, not a unit-specific nominal
    span. Exact apparent size can vary under perspective/zoom.
    """
    extent = ax.get_window_extent()
    viewport_points = min(extent.width, extent.height) * 72.0 / ax.figure.dpi
    diameter_points = (radius / visible_half_extent) * viewport_points
    return diameter_points ** 2


def _apply_framing(ax, framing: SceneFraming) -> None:
    """Apply a world-space cubic framing without imposing an absolute floor."""
    center = framing.center
    half = framing.half_extent
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


class MatplotlibBackend:
    """Render robot kinematic diagrams using Matplotlib 3D."""

    @staticmethod
    def render(
        scene_data: SceneData,
        ax=None,
        figsize=(10, 8),
        view_init=(30, 30),
        style: VisualizationStyle | None = None,
        layout: VisualizationLayout | None = None,
        framing: SceneFraming | None = None,
    ):
        """Render one robot configuration on a Matplotlib 3D axis."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        if style is None:
            style = VisualizationStyle()

        if layout is None:
            layout = resolve_layout([scene_data])
        if framing is None:
            framing = resolve_framing([scene_data], layout=layout, style=style)

        joints = scene_data.joints
        frames = scene_data.frames

        frame_scale = (
            layout.frame_length if style.frame_scale is None else style.frame_scale
        )
        # Preserve explicit Matplotlib marker-size semantics: s = size * 100
        # in points squared. Automatic sizes use the nominal world-space
        # radii resolved in the common VisualizationLayout.
        joint_area = (
            _marker_area_from_world_radius(
                ax, layout.joint_radius, framing.half_extent
            )
            if style.joint_size is None else style.joint_size * 100
        )
        base_area = (
            _marker_area_from_world_radius(
                ax, layout.base_radius, framing.half_extent
            )
            if style.base_size is None else style.base_size * 100
        )

        if style.show_links:
            if scene_data.dh_segments:
                # Draw the evaluated DH contributions directly instead of a
                # diagonal between consecutive frame origins. Retain zero
                # length segments in SceneData, but do not draw empty lines.
                for segment in scene_data.dh_segments:
                    if np.array_equal(segment.start, segment.end):
                        continue
                    ax.plot(
                        (segment.start[0], segment.end[0]),
                        (segment.start[1], segment.end[1]),
                        (segment.start[2], segment.end[2]),
                        color=style.link_color,
                        linewidth=style.link_linewidth,
                    )
            else:
                # Legacy scenes constructed without evaluated DH geometry.
                xs = [joint[0] for joint in joints]
                ys = [joint[1] for joint in joints]
                zs = [joint[2] for joint in joints]
                ax.plot(
                    xs,
                    ys,
                    zs,
                    "o-",
                    color=style.link_color,
                    linewidth=style.link_linewidth,
                    markersize=0,
                )

        if style.show_joints:
            for index, position in enumerate(joints):
                if index == 0 and not style.show_base:
                    continue

                color = style.base_color if index == 0 else style.joint_color
                marker_area = base_area if index == 0 else joint_area
                ax.scatter(
                    position[0],
                    position[1],
                    position[2],
                    color=color,
                    s=marker_area,
                    zorder=10,
                )

        if style.show_frames:
            for frame in frames:
                origin = frame.position
                for axis_name, color in zip(("x", "y", "z"), ("r", "g", "b")):
                    direction = getattr(frame, axis_name)
                    ax.quiver(
                        origin[0],
                        origin[1],
                        origin[2],
                        direction[0],
                        direction[1],
                        direction[2],
                        color=color,
                        length=frame_scale,
                        arrow_length_ratio=0.2,
                    )

        _apply_framing(ax, framing)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=view_init[0], azim=view_init[1])

        if not style.show_grid:
            ax.grid(False)

        return fig, ax

    @staticmethod
    def animate(
        scene_data_list: list[SceneData],
        interval=100,
        figsize=(10, 8),
        style: VisualizationStyle | None = None,
        layout: VisualizationLayout | None = None,
        framing: SceneFraming | None = None,
        **kwargs,
    ):
        """
        Create a Matplotlib animation.

        Notes
        -----
        Keep a reference to the returned ``FuncAnimation`` until it has been
        displayed or saved; otherwise it may be garbage-collected prematurely.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if not scene_data_list:
            raise ValueError(
                "scene_data_list must contain at least one scene."
            )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if style is None:
            style = VisualizationStyle()
        if layout is None:
            layout = resolve_layout(scene_data_list)

        if framing is None:
            framing = resolve_framing(
                scene_data_list, layout=layout, style=style
            )

        trajectory_points = np.array(
            _extract_end_effector_trajectory(scene_data_list),
            dtype=float,
        )

        def apply_global_limits():
            _apply_framing(ax, framing)

        def init():
            apply_global_limits()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            return (ax,)

        def update(frame_index):
            ax.clear()
            MatplotlibBackend.render(
                scene_data_list[frame_index],
                style=style,
                ax=ax,
                layout=layout,
                framing=framing,
                **kwargs,
            )

            if style.show_trajectory and trajectory_points.size > 0:
                if style.trajectory_mode == "trace":
                    points = trajectory_points[:frame_index + 1]
                else:
                    points = trajectory_points

                if points.size > 0:
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        color=style.trajectory_color,
                        linewidth=style.trajectory_linewidth,
                    )

            apply_global_limits()
            return (ax,)

        return FuncAnimation(
            fig,
            update,
            frames=len(scene_data_list),
            init_func=init,
            interval=interval,
            blit=False,
        )
