"""Convert symbolic robot configurations into numerical scene data."""

import numpy as np

from moro.core import Robot
from moro.util import sympy_matrix_to_numpy_float

from .data import FrameData, SceneData


def evaluate_robot(robot: Robot, num_vals: dict) -> SceneData:
    """
    Evaluate a robot at a numerical configuration.

    Parameters
    ----------
    robot : Robot
        Serial-link robot instance.
    num_vals : dict
        Mapping from symbolic variables to numerical values.

    Returns
    -------
    SceneData
        Numerical joint positions and frame transformations ready to render.
    """
    joints: list[np.ndarray] = []
    frames: list[FrameData] = []

    base_frame = FrameData(np.eye(4))
    frames.append(base_frame)
    joints.append(base_frame.position.copy())

    for i in range(1, robot.dof + 1):
        Ti = robot.T_i0(i).subs(num_vals)
        Ti_num = sympy_matrix_to_numpy_float(Ti)
        frame = FrameData(Ti_num)

        frames.append(frame)
        joints.append(frame.position.copy())

    all_coords = [coordinate for joint in joints for coordinate in joint]
    max_coord = max(abs(coordinate) for coordinate in all_coords) if all_coords else 1.0
    dimension = max(max_coord * 1.5, 1.0)

    return SceneData(
        joints=joints,
        frames=frames,
        dimension=dimension,
    )
