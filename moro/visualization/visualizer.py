"""Public orchestrator for robot visualization."""

from collections.abc import Mapping

import numpy as np

from moro.core import Robot

from .evaluation import evaluate_robot
from .matplotlib_backend import MatplotlibBackend
from .threejs_backend import ThreeJSBackend


def _normalize_configuration(robot, values):
    """Normalize a mapping or numerical joint vector for visualization."""
    if isinstance(values, Mapping):
        return values

    try:
        raw = np.asarray(values)
    except Exception as exc:
        raise ValueError(
            "A numerical robot configuration must be a one-dimensional "
            "vector with one finite real value per DOF."
        ) from exc

    if np.iscomplexobj(raw):
        raise ValueError(
            "A numerical robot configuration must contain only real values."
        )

    try:
        q = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "A numerical robot configuration must contain finite real values."
        ) from exc

    if q.ndim != 1 or q.size != robot.dof:
        raise ValueError(
            "A numerical robot configuration must be one-dimensional with "
            f"exactly robot.dof={robot.dof} values."
        )
    if not np.all(np.isfinite(q)):
        raise ValueError(
            "A numerical robot configuration must contain only finite values."
        )

    return dict(zip(robot.qs, q.tolist()))


def _normalize_configurations(robot, values):
    """Normalize a non-empty sequence of robot configurations."""
    if isinstance(values, np.ndarray):
        raw = values
        if np.iscomplexobj(raw):
            raise ValueError(
                "Numerical robot configurations must contain only real values."
            )
        if raw.ndim != 2:
            raise ValueError(
                "Numerical animation data must have shape (N, robot.dof)."
            )
        if raw.shape[0] == 0:
            raise ValueError(
                "num_vals_list must contain at least one configuration."
            )
        if raw.shape[1] != robot.dof:
            raise ValueError(
                "Numerical animation data must have shape "
                f"(N, {robot.dof})."
            )
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Numerical robot configurations must contain finite real values."
            ) from exc
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Numerical robot configurations must contain only finite values."
            )
        return [
            dict(zip(robot.qs, row.tolist()))
            for row in matrix
        ]

    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            "num_vals_list must be an iterable of robot configurations."
        ) from exc

    if len(items) == 0:
        raise ValueError(
            "num_vals_list must contain at least one configuration."
        )

    return [
        _normalize_configuration(robot, item)
        for item in items
    ]


class RobotVisualizer:
    """Render a :class:`moro.core.Robot` using the available backends."""

    def __init__(self, robot: Robot):
        if not isinstance(robot, Robot):
            raise TypeError(
                f"Expected a Robot instance, got {type(robot).__name__}."
            )
        self.robot = robot

    def plot(self, num_vals, backend="matplotlib", **kwargs):
        """Render one robot configuration from a mapping or joint vector."""
        configuration = _normalize_configuration(self.robot, num_vals)
        scene_data = evaluate_robot(self.robot, configuration)

        if backend == "matplotlib":
            return MatplotlibBackend.render(scene_data, **kwargs)
        if backend == "threejs":
            return ThreeJSBackend.render(scene_data, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )

    def animate(self, num_vals_list, backend="matplotlib", **kwargs):
        """Animate mappings or numerical joint vectors in processing order."""
        configurations = _normalize_configurations(
            self.robot,
            num_vals_list,
        )

        scene_data_list = [
            evaluate_robot(self.robot, num_vals)
            for num_vals in configurations
        ]

        if backend == "matplotlib":
            return MatplotlibBackend.animate(scene_data_list, **kwargs)
        if backend == "threejs":
            return ThreeJSBackend.animate(scene_data_list, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )
