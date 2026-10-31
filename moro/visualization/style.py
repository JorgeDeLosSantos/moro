"""Backend-independent visualization style configuration."""

from dataclasses import dataclass
import math
from typing import Literal


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
    joint_representation : str
        'simple' keeps frame-origin spheres; 'kinematic' draws R/P axis symbols.
    joint_scale : float
        Finite positive dimensionless factor for kinematic joint symbols.
    joint_appearance : str
        Three.js experimental R/P appearance: 'schematic', 'solid' or 'hybrid'.
        Matplotlib supports 'schematic' only for now.
    link_color : str
        Color used for links.
    joint_color : str
        Color used for joints.
    base_color : str
        Color used for the base joint.
    frame_scale : float or None
        Length of frame axes in robot units. If None, use the shared layout.
    joint_size : float or None
        Explicit size: Matplotlib marker area factor (s=size*100 in pt²),
        or Three.js world-space radius. If None, use the backend adapter.
    base_size : float or None
        Base marker size with the same backend-specific semantics as
        joint_size. If None, use the backend adapter.
    link_linewidth : float
        Link thickness in Matplotlib points; Three.js uses it as a
        multiplicative factor relative to its nominal link radius (value/3).
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

    # Appended (rather than inserted before colors) to preserve the
    # dataclass's previously supported positional-argument order.
    joint_representation: Literal["simple", "kinematic"] = "simple"
    joint_scale: float = 1.0
    joint_appearance: Literal["schematic", "solid", "hybrid"] = "schematic"

    def __post_init__(self):
        if self.joint_appearance not in {"schematic", "solid", "hybrid"}:
            raise ValueError(
                "joint_appearance must be 'schematic', 'solid' or 'hybrid'."
            )
        if self.joint_representation not in {'simple', 'kinematic'}:
            raise ValueError("joint_representation must be 'simple' or 'kinematic'.")
        if isinstance(self.joint_scale, bool):
            raise ValueError('joint_scale must be a finite positive number.')
        try:
            scale = float(self.joint_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('joint_scale must be a finite positive number.') from exc
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('joint_scale must be a finite positive number.')
        self.joint_scale = scale
        if self.trajectory_mode not in {"full", "trace"}:
            raise ValueError(
                "trajectory_mode must be either 'full' or 'trace'"
            )
