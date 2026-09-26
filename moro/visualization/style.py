"""Backend-independent visualization style configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class VisualizationStyle:
    """
    Style configuration for robot visualization.

    Parameters
    ----------
    show_frames : bool
        Draw coordinate frames at each joint.
    show_links : bool
        Draw links connecting consecutive joints.
    show_joints : bool
        Draw joint markers.
    show_base : bool
        Highlight the base joint when joints are visible.
    show_grid : bool
        Draw the reference grid.
    link_color : str
        Color used for links.
    joint_color : str
        Color used for joints.
    base_color : str
        Color used for the base joint.
    frame_scale : float or None
        Length of frame axes. If None, derive it from the scene dimension.
    joint_size : float or None
        Joint size. If None, derive it from the scene dimension.
    base_size : float or None
        Base size. If None, derive it from the scene dimension.
    link_linewidth : float
        Link thickness. Three.js interprets it as a relative thickness factor.
    show_trajectory : bool
        Draw end-effector Cartesian trajectory in animations.
    trajectory_color : str
        Color used for the trajectory line.
    trajectory_linewidth : float
        Trajectory line thickness.
    trajectory_mode : str
        Trajectory display mode: ``"full"`` or ``"trace"``.
    """

    show_frames: bool = True
    show_links: bool = True
    show_joints: bool = True
    show_base: bool = True
    show_grid: bool = True

    link_color: str = "#778877"
    joint_color: str = "#ff1493"
    base_color: str = "#ff00ff"

    frame_scale: float | None = None
    joint_size: float | None = None
    base_size: float | None = None

    link_linewidth: float = 3
    show_trajectory: bool = False
    trajectory_color: str = "#1565c0"
    trajectory_linewidth: float = 2
    trajectory_mode: str = "full"

    def __post_init__(self):
        if self.trajectory_mode not in {"full", "trace"}:
            raise ValueError(
                "trajectory_mode must be either 'full' or 'trace'"
            )
