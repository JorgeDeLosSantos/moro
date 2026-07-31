"""Matplotlib visualization backend."""

import numpy as np

from .data import SceneData
from .style import VisualizationStyle
from .threejs_backend import _extract_end_effector_trajectory


class MatplotlibBackend:
    """Render robot kinematic diagrams using Matplotlib 3D."""

    @staticmethod
    def render(
        scene_data: SceneData,
        ax=None,
        figsize=(10, 8),
        view_init=(30, 30),
        style: VisualizationStyle | None = None,
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

        joints = scene_data.joints
        frames = scene_data.frames
        dim = scene_data.dimension

        frame_scale = dim / 5 if style.frame_scale is None else style.frame_scale
        joint_size = (
            max(dim * 0.025, 0.1)
            if style.joint_size is None
            else style.joint_size
        )
        base_size = (
            max(dim * 0.035, 0.15)
            if style.base_size is None
            else style.base_size
        )

        if style.show_links:
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
                size = base_size if index == 0 else joint_size
                ax.scatter(
                    position[0],
                    position[1],
                    position[2],
                    color=color,
                    s=size * 100,
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

        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_zlim(-dim, dim)
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

        dim = max(scene.dimension for scene in scene_data_list)
        trajectory_points = np.array(
            _extract_end_effector_trajectory(scene_data_list),
            dtype=float,
        )

        bounds_points: list[np.ndarray] = []
        for scene in scene_data_list:
            if scene.joints:
                bounds_points.append(np.asarray(scene.joints, dtype=float))
            if scene.frames:
                bounds_points.append(np.asarray([
                    frame.position
                    for frame in scene.frames
                ], dtype=float))

        if style.show_trajectory and trajectory_points.size > 0:
            bounds_points.append(trajectory_points)

        if bounds_points:
            stacked = np.vstack(bounds_points)
            mins = stacked.min(axis=0)
            maxs = stacked.max(axis=0)
            center = (mins + maxs) / 2
            half_extent = max(np.max(maxs - mins) / 2, dim)
        else:
            center = np.zeros(3, dtype=float)
            half_extent = dim

        def apply_global_limits():
            ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            ax.set_zlim(center[2] - half_extent, center[2] + half_extent)

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
