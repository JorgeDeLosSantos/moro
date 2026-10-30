"""Convert symbolic robot configurations into numerical scene data."""

import numpy as np
import sympy as sp

from moro.core import Robot
from moro.util import sympy_matrix_to_numpy_float

from .data import DHSegmentData, FrameData, SceneData


def _evaluate_dh_translation(value, num_vals: dict, row_index: int, name: str) -> float:
    """Evaluate a signed DH translation to a finite real scalar."""
    try:
        result = float(sp.sympify(value).subs(num_vals))
        if not np.isfinite(result):
            raise ValueError("non-finite DH translation")
    except (TypeError, ValueError, OverflowError, AttributeError, sp.SympifyError) as exc:
        raise ValueError(
            f"DH row {row_index} parameter '{name}' must evaluate "
            "to a finite real number."
        ) from exc
    return result


def _build_dh_segments(
    robot: Robot,
    num_vals: dict,
    frames: list[FrameData],
) -> list[DHSegmentData]:
    """Build the signed d/a translation segments from evaluated DH frames.

    Endpoints are independent copies in base/world coordinates. The helper
    does not recompute homogeneous transformations.
    """
    segments: list[DHSegmentData] = []

    for i, (a_i, _, d_i, _) in enumerate(robot.dh_parameters, start=1):
        # Check both translations, even though the final frame already
        # determines the endpoint of the a-segment.
        _evaluate_dh_translation(a_i, num_vals, i, "a")
        d_value = _evaluate_dh_translation(d_i, num_vals, i, "d")

        previous_frame = frames[i - 1]
        current_frame = frames[i]

        p_prev = np.array(previous_frame.position, dtype=float, copy=True)
        p_mid = p_prev + d_value * previous_frame.z
        p_end = np.array(current_frame.position, dtype=float, copy=True)

        segments.append(
            DHSegmentData(
                row_index=i,
                kind="d",
                start=p_prev.copy(),
                end=p_mid.copy(),
            )
        )
        segments.append(
            DHSegmentData(
                row_index=i,
                kind="a",
                start=p_mid.copy(),
                end=p_end.copy(),
            )
        )

    return segments


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
        Numerical joint positions, frame transformations and ordered
        DH translation segments in the base frame, ready for rendering.
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
        dh_segments=_build_dh_segments(robot, num_vals, frames),
    )
