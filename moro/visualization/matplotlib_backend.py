"""Matplotlib visualization backend."""

from .data import SceneData
from .style import VisualizationStyle


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

        dim = scene_data_list[0].dimension

        def init():
            ax.set_xlim(-dim, dim)
            ax.set_ylim(-dim, dim)
            ax.set_zlim(-dim, dim)
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
            return (ax,)

        return FuncAnimation(
            fig,
            update,
            frames=len(scene_data_list),
            init_func=init,
            interval=interval,
            blit=False,
        )
