"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots.
This library has been designed, mainly, for academic and research purposes,
using SymPy as base library.

visualization module provides tools for rendering the kinematic diagram
of a serial robot using different backends (matplotlib, Three.js).
"""
import json
import re
import uuid

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np

from moro.core import Robot
from moro.util import sympy_matrix_to_numpy_float


__all__ = [
    "RobotVisualizer",
    "MatplotlibBackend",
    "ThreeJSBackend",
    "evaluate_robot",
    "SceneData",
    "FrameData",
    "VisualizationStyle",
    "_replace_placeholders",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SceneData:
    """
    Evaluated data of a robot configuration.
    """
    joints: list[np.ndarray]
    frames: list["FrameData"]
    dimension: float


@dataclass(slots=True)
class FrameData:
    """
    Numerical homogeneous transformation of one robot frame.
    """

    T: np.ndarray

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def rotation(self):
        return self.T[:3, :3]

    @property
    def x(self):
        return self.rotation[:, 0]

    @property
    def y(self):
        return self.rotation[:, 1]

    @property
    def z(self):
        return self.rotation[:, 2]


@dataclass(slots=True)
class VisualizationStyle:
    """
    Style configuration for robot visualization.

    show_frames : bool
        Draw coordinate frames at each joint.
    show_links : bool
        Draw links as lines connecting joints.
    show_joints : bool
        Draw spheres at joint positions.
    show_base : bool
        Highlight the base joint.
    show_grid : bool
        Show the 3D grid.
    frame_scale : float or None
        Length of the frame axes. If None, derived from scene dimension.
    link_color : str
        Color for the links.
    link_linewidth : float
        Line width for links.
    joint_color : str
        Color for joint spheres.
    base_color : str
        Color for the base joint sphere.
    joint_size : float or None
        Size of joint spheres. If None, derived from scene dimension.
    base_size : float or None
        Size of base sphere. If None, derived from scene dimension.
    """

    # Visibility
    show_frames: bool = True
    show_links: bool = True
    show_joints: bool = True
    show_base: bool = True
    show_grid: bool = True

    # Colors
    link_color: str = "#778877"
    joint_color: str = "#ff1493"
    base_color: str = "#ff00ff"

    # Sizes
    frame_scale: float | None = None
    joint_size: float | None = None
    base_size: float | None = None

    # Line styles
    link_linewidth: float = 3


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_robot(robot, num_vals):
    """
    Evaluate the robot at a given numerical configuration and return a
    :class:`SceneData` instance containing joint positions and frame
    orientations.

    Parameters
    ----------
    robot : Robot
        A serial-link robot instance.
    num_vals : dict
        Dictionary mapping symbolic joint variables (or any other symbols)
        to numerical values, e.g. ``{q1: 0.5, q2: 1.2}``.

    Returns
    -------
    SceneData
        Evaluated scene data ready for rendering.
    """
    joints = []
    frames = []

    # Base frame (index 0)
    base_frame = FrameData(np.eye(4))
    frames.append(base_frame)
    joints.append(base_frame.position.copy())

    # Link frames
    for i in range(1, robot.dof + 1):
        Ti = robot.T_i0(i).subs(num_vals)
        Ti_num = sympy_matrix_to_numpy_float(Ti)

        frame = FrameData(Ti_num)

        frames.append(frame)
        joints.append(frame.position.copy())

    # Characteristic dimension for view scaling
    all_coords = [c for joint in joints for c in joint]
    max_coord = max(abs(c) for c in all_coords) if all_coords else 1.0
    dimension = max(max_coord * 1.5, 1.0)

    return SceneData(joints=joints, frames=frames, dimension=dimension)



_PLACEHOLDER_PATTERN = re.compile(r"__[A-Z][A-Z0-9_]*__")


def _replace_placeholders(
    template: str,
    replacements: dict[str, Any],
) -> str:
    """
    Replace ``__PLACEHOLDER__`` tokens in *template* with the corresponding
    values from *replacements*.

    Parameters
    ----------
    template : str
        String containing ``__PLACEHOLDER__`` tokens.
    replacements : dict[str, Any]
        Mapping of placeholder names (without the leading/trailing underscores)
        to their replacement values.

    Returns
    -------
    str
        Template with all placeholders replaced.

    Raises
    ------
    ValueError
        If a replacement key does not match any placeholder in the template,
        or if there are unresolved placeholders remaining after substitution.
    """
    html = template

    for name, value in replacements.items():
        placeholder = f"__{name.upper()}__"

        if placeholder not in html:
            raise ValueError(
                f"Placeholder {placeholder!r} was not found in template."
            )

        html = html.replace(placeholder, str(value))

    unresolved = sorted(set(_PLACEHOLDER_PATTERN.findall(html)))

    if unresolved:
        raise ValueError(
            f"Unresolved placeholders in template: {unresolved}"
        )

    return html


def _render_html_template(
    template_name: str,
    replacements: dict[str, Any],
) -> str:
    """
    Load a template file from the ``moro.templates`` package and replace
    its ``__PLACEHOLDER__`` tokens.

    Parameters
    ----------
    template_name : str
        Name of the template file (e.g. ``"threejs_viewer.html"``).
    replacements : dict[str, Any]
        Mapping of placeholder names to replacement values.

    Returns
    -------
    str
        Fully rendered template string.
    """
    template = (
        files("moro.templates")
        .joinpath(template_name)
        .read_text(encoding="utf-8")
    )

    return _replace_placeholders(template, replacements)





def _scene_to_payload(scene_data: SceneData) -> dict:
    """
    Convert scene data into a JSON-serializable dictionary.
    """
    return {
        "joints": [
            np.asarray(joint, dtype=float).tolist()
            for joint in scene_data.joints
        ],
        "frames": [
            {
                "position": frame.position.tolist(),
                "x": frame.x.tolist(),
                "y": frame.y.tolist(),
                "z": frame.z.tolist(),
            }
            for frame in scene_data.frames
        ],
        "dimension": float(scene_data.dimension),
    }


def _scenes_to_payload(scene_data_list: list[SceneData]) -> list[dict]:
    """
    Convert multiple scenes into JSON-serializable dictionaries.
    """
    return [_scene_to_payload(scene) for scene in scene_data_list]



# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------

class MatplotlibBackend:
    """
    Backend that renders the robot kinematic diagram using Matplotlib 3D.
    """

    @staticmethod
    def render(
        scene_data,
        ax=None,
        figsize=(10, 8),
        view_init=(30, 30),
        style=None
    ):
        """
        Render the robot scene on a Matplotlib 3D axis.

        Parameters
        ----------
        scene_data : SceneData
            Evaluated robot data.
        ax : Axes3D or None
            Existing axis to draw on. If None, a new figure and axis are created.
        figsize : tuple
            Figure size in inches.
        view_init : tuple
            Initial view angles ``(elev, azim)``.
        style : VisualizationStyle or None
            Style configuration for the visualization.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : mpl_toolkits.mplot3d.Axes3D
        """
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

        # Derived sizes
        if style.frame_scale is None:
            frame_scale = dim / 5
        else:
            frame_scale = style.frame_scale

        if style.joint_size is None:
            joint_size = max(dim * 0.025, 0.1)
        else:
            joint_size = style.joint_size

        if style.base_size is None:
            base_size = max(dim * 0.035, 0.15)
        else:
            base_size = style.base_size

        # --- Links ---
        if style.show_links:
            xs = [j[0] for j in joints]
            ys = [j[1] for j in joints]
            zs = [j[2] for j in joints]
            ax.plot(
                xs, ys, zs,
                "o-",
                color=style.link_color,
                linewidth=style.link_linewidth,
                markersize=0,  # we draw custom joint spheres
            )

        # --- Joints ---
        if style.show_joints:
            for idx, pos in enumerate(joints):
                if idx == 0 and not style.show_base:
                    continue
                color = style.base_color if idx == 0 else style.joint_color
                size = base_size if idx == 0 else joint_size
                ax.scatter(
                    pos[0], pos[1], pos[2],
                    color=color,
                    s=size * 100,  # scatter uses points^2
                    zorder=10,
                )

        # --- Frames ---
        if style.show_frames:
            for frame in frames:
                origin = frame.position
                for axis_name, color in zip(("x", "y", "z"), ("r", "g", "b")):
                    direction = getattr(frame, axis_name)
                    ax.quiver(
                        origin[0], origin[1], origin[2],
                        direction[0], direction[1], direction[2],
                        color=color,
                        length=frame_scale,
                        arrow_length_ratio=0.2,
                    )

        # --- Axes setup ---
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
        scene_data_list,
        interval=100,
        figsize=(10, 8),
        style=None,
        **kwargs,
    ):
        """
        Create a matplotlib animation from a list of scene data.

        Parameters
        ----------
        scene_data_list : list of SceneData
            Sequence of evaluated robot configurations.
        interval : int
            Delay between frames in milliseconds.
        figsize : tuple
            Figure size in inches.
        **kwargs
            Additional arguments forwarded to :meth:`render`.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
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

        # Determine scene dimension from the first frame
        dim = scene_data_list[0].dimension

        def init():
            ax.set_xlim(-dim, dim)
            ax.set_ylim(-dim, dim)
            ax.set_zlim(-dim, dim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            return (ax,)

        def update(frame_idx):
            ax.clear()
            MatplotlibBackend.render(
                scene_data_list[frame_idx],
                style=style,
                ax=ax,
                **kwargs,
            )
            return (ax,)

        anim = FuncAnimation(
            fig, update, frames=len(scene_data_list),
            init_func=init, interval=interval, blit=False,
        )
        
        return anim


# ---------------------------------------------------------------------------
# Three.js backend
# ---------------------------------------------------------------------------

class ThreeJSBackend:
    """
    Backend that renders the robot kinematic diagram using Three.js.

    The output is an HTML string that can be displayed in a Jupyter notebook
    (via ``IPython.display.HTML``) or saved to a standalone file.

    Controls (CAD-like):
        - Scroll wheel: orbit (rotate scene)
        - Left button: pan
        - Right button: pan
        - Scroll wheel: zoom
    """

    @staticmethod
    def render(
        scene_data: SceneData,
        width: int = 800,
        height: int = 600,
        style: VisualizationStyle | None = None,
    ):
        """
        Generate an interactive Three.js HTML view.

        Parameters
        ----------
        scene_data : SceneData
            Evaluated robot data.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.
        style : VisualizationStyle or None
            Visualization style. Currently reserved for future Three.js
            styling support.

        Returns
        -------
        IPython.display.HTML
            HTML display object containing the interactive view.
        """
        from IPython.display import HTML

        if style is None:
            style = VisualizationStyle()

        unique_id = uuid.uuid4().hex[:8]
        payload = _scene_to_payload(scene_data)

        html = _render_html_template(
            "threejs_viewer.html",
            {
                "unique_id": unique_id,
                "width": width,
                "height": height,
                "robot_data": json.dumps(payload),
            },
        )

        return HTML(html)
    

    @staticmethod
    def animate(
        scene_data_list: list[SceneData],
        width: int = 800,
        height: int = 600,
        style: VisualizationStyle | None = None,
    ):
        """
        Create an interactive Three.js animation.

        Parameters
        ----------
        scene_data_list : list of SceneData
            Sequence of evaluated robot configurations.
        width : int
            Canvas width in pixels.
        height : int
            Canvas height in pixels.
        style : VisualizationStyle or None
            Visualization style. Currently reserved for future Three.js
            styling support.

        Returns
        -------
        IPython.display.HTML
            HTML display object containing the interactive animation.
        """
        from IPython.display import HTML

        if style is None:
            style = VisualizationStyle()

        if not scene_data_list:
            raise ValueError(
                "scene_data_list must contain at least one scene."
            )

        unique_id = uuid.uuid4().hex[:8]
        payloads = _scenes_to_payload(scene_data_list)

        html = _render_html_template(
            "threejs_animation.html",
            {
                "unique_id": unique_id,
                "width": width,
                "height": height,
                "frames_data": json.dumps(payloads),
                "last_frame": len(payloads) - 1,
            },
        )

        return HTML(html)



# ---------------------------------------------------------------------------
# Main visualizer (orchestrator)
# ---------------------------------------------------------------------------

class RobotVisualizer:
    """
    Orchestrator for rendering a :class:`Robot` in different visualization
    backends.

    Parameters
    ----------
    robot : Robot
        A serial-link robot instance.
    """

    def __init__(self, robot):
        if not isinstance(robot, Robot):
            raise TypeError(
                f"Expected a Robot instance, got {type(robot).__name__}."
            )
        self.robot = robot

    # ---- Single-configuration plot ----------------------------------------

    def plot(self, num_vals, backend="matplotlib", **kwargs):
        """
        Render the robot kinematic diagram at a given configuration.

        Parameters
        ----------
        num_vals : dict
            Dictionary mapping symbolic joint variables to numerical values,
            e.g. ``{q1: 0.5, q2: 1.2}``.
        backend : str
            Visualization backend. ``"matplotlib"`` (default) or ``"threejs"``.
        **kwargs
            Additional arguments forwarded to the backend's ``render`` method.

        Returns
        -------
        Depends on the backend:
            - ``"matplotlib"`` : ``(fig, ax)`` tuple.
            - ``"threejs"``    : HTML string.
        """
        scene_data = evaluate_robot(self.robot, num_vals)

        if backend == "matplotlib":
            return MatplotlibBackend.render(scene_data, **kwargs)
        elif backend == "threejs":
            return ThreeJSBackend.render(scene_data, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: 'matplotlib', 'threejs'."
            )


    # ---- Animation --------------------------------------------------------

    def animate(self, num_vals_list, backend="matplotlib", **kwargs):
        """
        Create an animation from a sequence of robot configurations.

        Parameters
        ----------
        num_vals_list : list of dict
            Each element is a ``dict`` mapping symbolic joint variables to
            numerical values for one frame of the animation.
        backend : str
            ``"matplotlib"`` (returns a FuncAnimation) or ``"threejs"``
            (returns an IPython HTML display).
        **kwargs
            Additional arguments forwarded to the backend's ``animate`` method.

        Returns
        -------
        Depends on the backend:
            - ``"matplotlib"`` : ``matplotlib.animation.FuncAnimation``.
            - ``"threejs"``    : ``IPython.display.HTML``.
        """
        scene_data_list = [
            evaluate_robot(self.robot, nv) for nv in num_vals_list
        ]

        if backend == "matplotlib":
            return MatplotlibBackend.animate(scene_data_list, **kwargs)
        elif backend == "threejs":
            return ThreeJSBackend.animate(scene_data_list, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: 'matplotlib', 'threejs'."
            )