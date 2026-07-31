"""Public orchestrator for robot visualization."""

from moro.core import Robot

from .evaluation import evaluate_robot
from .matplotlib_backend import MatplotlibBackend
from .threejs_backend import ThreeJSBackend


class RobotVisualizer:
    """Render a :class:`moro.core.Robot` using the available backends."""

    def __init__(self, robot: Robot):
        if not isinstance(robot, Robot):
            raise TypeError(
                f"Expected a Robot instance, got {type(robot).__name__}."
            )
        self.robot = robot

    def plot(self, num_vals: dict, backend="matplotlib", **kwargs):
        """Render one robot configuration."""
        scene_data = evaluate_robot(self.robot, num_vals)

        if backend == "matplotlib":
            return MatplotlibBackend.render(scene_data, **kwargs)
        if backend == "threejs":
            return ThreeJSBackend.render(scene_data, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )

    def animate(self, num_vals_list, backend="matplotlib", **kwargs):
        """Animate a sequence of robot configurations."""
        if not num_vals_list:
            raise ValueError(
                "num_vals_list must contain at least one configuration."
            )

        scene_data_list = [
            evaluate_robot(self.robot, num_vals)
            for num_vals in num_vals_list
        ]

        if backend == "matplotlib":
            return MatplotlibBackend.animate(scene_data_list, **kwargs)
        if backend == "threejs":
            return ThreeJSBackend.animate(scene_data_list, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )
