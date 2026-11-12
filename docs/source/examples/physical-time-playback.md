# Physical-Time Animation Playback in Three.js

This Jupyter/Colab-ready example demonstrates **T-06**: visualizing
robot joint configurations at their real, possibly nonuniform, sample
times. Unlike the older fixed-`interval` player, timestamped playback
uses the time vector produced by the trajectory generator.

These exercises use the physical-time playback API introduced in Moro 0.7.0.
They use a fully numerical manipulator model, so no additional geometric
parameter substitutions are needed. For a model with symbolic lengths,
the API-001 `parameters=` argument lets you supply those fixed values
once, alongside `times=` and the numerical joint matrix. See
[Symbolic DH Parameters in Robot Visualization](symbolic-visualization-parameters.md).

## 1. Prepare a planar RR manipulator

```python
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from moro import Robot
from moro.abc import q1, q2
from moro.trajectory import (
    joint_trapezoidal_trajectory,
    joint_waypoint_trajectory,
)
from moro.visualization import RobotVisualizer

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
)
viz = RobotVisualizer(robot)
```

`RobotVisualizer.animate()` continues to accept a numerical matrix of
shape `(samples, robot.dof)`, with or without explicit timestamps. The
visualization module needs neither a `JointTrajectory` instance nor an
import of the trajectory generator.

## 2. Compare uniform frames and nonuniform physical time

Specify two nonuniformly spaced motion segments, sampling more
densely in some parts of the trajectory than in others:

```python
waypoints = np.array([
    [0.0, 0.0],
    [0.65, -0.3],
    [1.2, 0.4],
])
waypoint_times = [0.0, 1.0, 2.6]
sample_times = np.array([
    0.0, 0.07, 0.20, 0.50, 1.0,
    1.05, 1.40, 1.70, 2.00, 2.30, 2.60,
])

motion = joint_waypoint_trajectory(
    waypoints,
    waypoint_times,
    method="cubic",
    derivative_policy="shape_preserving",
    sample_times=sample_times,
)

assert motion.q.shape == (11, 2)
np.testing.assert_allclose(motion.t, sample_times)
assert not np.allclose(np.diff(motion.t), np.diff(motion.t)[0])
```

With the legacy player, **every frame has the same display interval**,
irrespective of the differences between adjacent physical times:

```python
legacy = viz.animate(motion.q, interval=150)
display(legacy)
```

With T-06, each sample is shown at its correct *relative physical
timestamp*. A configuration remains displayed until the next sample
is due; Moro does **not** interpolate robot configurations in the
browser:

```python
physical = viz.animate(
    motion.q,
    times=motion.t,
    playback_speed=1.0,
)
display(physical)
```

The physical-time playback duration is 2.6 seconds at 1.0× speed.
For the timestamps `[0.0, 0.07, 0.20, 0.50, ...]`, the initial
configuration is displayed briefly, whereas the frame at 0.20 seconds
is held for 0.30 seconds.

To see the sampling distribution explicitly:

```python
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(motion.t, np.zeros_like(motion.t), "|", markersize=24)
ax.set_xlabel("Physical time [s]")
ax.set_yticks([])
ax.set_xlim(motion.t[0] - 0.05, motion.t[-1] + 0.05)
ax.grid(axis="x")
plt.show()
```

## 3. Control the playback speed

`playback_speed` scales the passage of *physical time* in the player.
It does not alter any joint configuration or trajectory derivative:

```python
half_speed = viz.animate(
    motion.q, times=motion.t, playback_speed=0.5,
)
double_speed = viz.animate(
    motion.q, times=motion.t, playback_speed=2.0,
)

display(half_speed)
display(double_speed)
```

At half speed, a nominal 2.6-second motion takes approximately
5.2 seconds of *browser wall time* to play; at double speed it takes
approximately 1.3 seconds. Rendering, event-loop delays and browser
throttling mean these are not hard real-time guarantees.

The viewer contains an elapsed/total physical-time label, a speed
label, a frame selector, Play/Pause, camera presets and Fit View.
The frame selector always chooses an original **sample index**, not
a synthetic interpolated pose.

## 4. Use a trapezoidal trajectory directly

A trapezoidal profile already has uniformly spaced time samples, but
its duration comes from speed/acceleration limits rather than an
arbitrary frame interval:

```python
trapezoidal = joint_trapezoidal_trajectory(
    [0.0, 0.0],
    [1.2, -0.8],
    v_max=[0.6, 0.4],
    a_max=[1.0, 0.8],
    samples=161,
)
assert np.isclose(trapezoidal.duration, 2.6)
display(viz.animate(trapezoidal.q, times=trapezoidal.t))
```

For a nonzero absolute time origin, simply provide the original
trajectory timestamps; the browser subtracts the first timestamp:

```python
display(viz.animate(
    motion.q,
    times=motion.t + 7.0,
    playback_speed=1.5,
))
```

A constant shift in all timestamps has **no effect** on the relative
duration or temporal spacing.

## 5. Interactions and expected behavior

Use these checks when evaluating the widget interactively in Jupyter
or Colab:

1. Press Play. The time counter increases according to the physical
   timeline; the selected frame may remain fixed during long gaps.
2. Pause in the middle of a long gap. The counter and robot stop.
   Press Play again; playback continues rather than restarting.
3. While paused, drag the frame slider. The clock snaps to the
   corresponding timestamp and remains paused.
4. While playing, drag the frame slider backwards or forwards. The
   clock reanchors to the selected sample and continues.
5. Allow playback to finish. The last frame remains visible and Play
   becomes available again. Press Play to restart at the first frame.
6. Change camera presets, orbit the camera and use Fit View during
   playback; none of these should alter the playback clock.
7. Display two animations in one notebook and verify that their
   timers and controls are independent.

When a tab is inactive or rendering is delayed, the player may skip
some *displayed* frames to catch up to the physical-time clock. It
does not skip, modify or regenerate elements of the input trajectory.

## 6. The Matplotlib limitation is intentional

Matplotlib retains the existing fixed-interval `FuncAnimation` path:

```python
matplotlib_anim = viz.animate(
    motion.q,
    backend="matplotlib",
    interval=150,
)
```

This is **not** an exact representation of the nonuniform time grid.
Passing `times=motion.t` with `backend="matplotlib"` raises
`ValueError` with an explanation, rather than silently approximating
the requested physical-time playback.

Also, `times` and `interval` must not be specified together.
For background and the timing contract, see
[Visualization](../user-guide/visualization.md) and the
[T-06 design](https://github.com/JorgeDeLosSantos/moro/blob/master/docs/design/0.7.0/physical-time-playback.md).

**Teaching question:** Compare the two animations from Section 2.
How does holding every sample for 150 milliseconds change the student's
perception of velocity compared with respecting the original timestamps?
