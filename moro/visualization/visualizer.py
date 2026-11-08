"""Public orchestrator for robot visualization."""

from moro.core import Robot

from .evaluation import evaluate_robot
from .matplotlib_backend import MatplotlibBackend
from .playback import _normalize_playback_speed, _normalize_playback_times
from .substitutions import (
    _normalize_visualization_configurations,
    _normalize_visualization_substitutions,
)
from .threejs_backend import ThreeJSBackend


class RobotVisualizer:
    """Render a :class:`moro.core.Robot` using the available backends."""

    def __init__(self, robot: Robot):
        if not isinstance(robot, Robot):
            raise TypeError(
                f"Expected a Robot instance, got {type(robot).__name__}."
            )
        self.robot = robot

    def plot(self, configuration, backend="threejs", *, parameters=None, **kwargs):
        """Render one joint configuration with optional fixed DH parameters.

        configuration may be a numerical vector or a complete/partial
        symbolic substitution mapping. parameters supplies constant
        DH symbols separately, without mutating the robot or caller inputs.

        Returns IPython.display.HTML for Three.js, or (Figure, Axes3D)
        when backend='matplotlib' is requested explicitly.
        """
        substitutions = _normalize_visualization_substitutions(
            self.robot, configuration, parameters=parameters,
        )
        scene_data = evaluate_robot(self.robot, substitutions)

        if backend == "matplotlib":
            return MatplotlibBackend.render(scene_data, **kwargs)
        if backend == "threejs":
            return ThreeJSBackend.render(scene_data, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )

    def animate(
        self,
        configurations,
        backend="threejs",
        *,
        parameters=None,
        times=None,
        playback_speed=1.0,
        **kwargs,
    ):
        """Animate joint configurations with optional fixed DH parameters.

        Numerical arrays have shape (N, robot.dof); sequences of complete
        or partial symbolic substitution mappings are also supported.
        parameters is constant across frames and validated once.

        Legacy interval-based playback and the Matplotlib FuncAnimation
        return contract remain unchanged. With times=, Three.js uses the
        supplied physical timestamps instead of a fixed frame interval.
        """
        substitutions_list = _normalize_visualization_configurations(
            self.robot, configurations, parameters=parameters,
        )

        speed = _normalize_playback_speed(playback_speed)
        relative_times = None
        if times is None:
            if speed != 1.0:
                raise ValueError(
                    "playback_speed requires explicit times; "
                    "use interval for legacy frame playback."
                )
        else:
            if "interval" in kwargs:
                raise ValueError(
                    "times and interval are alternative playback controls; "
                    "provide only one."
                )
            relative_times = _normalize_playback_times(
                times, len(substitutions_list)
            )
            if backend == "matplotlib":
                raise ValueError(
                    "Matplotlib does not support physical-time playback "
                    "with times=; use the Three.js backend or omit times "
                    "and specify interval for Matplotlib."
                )

        scene_data_list = [
            evaluate_robot(self.robot, substitutions)
            for substitutions in substitutions_list
        ]

        if backend == "matplotlib":
            return MatplotlibBackend.animate(scene_data_list, **kwargs)
        if backend == "threejs":
            if relative_times is not None:
                kwargs["relative_times"] = relative_times
                kwargs["playback_speed"] = speed
            return ThreeJSBackend.animate(scene_data_list, **kwargs)

        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Available backends: 'matplotlib', 'threejs'."
        )
