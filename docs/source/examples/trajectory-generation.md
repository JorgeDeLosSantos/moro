# Polynomial Trajectory Generation

This example introduces linear, cubic, and quintic point-to-point trajectories and shows how their numerical arrays integrate with inverse kinematics and visualization.

## 1. Linear, cubic, and quintic comparison

```python
import numpy as np

from moro.trajectory import joint_trajectory

t = np.linspace(0.0, 2.0, 101)

linear = joint_trajectory(
    0.0,
    1.0,
    t,
    method="linear",
)

cubic = joint_trajectory(
    0.0,
    1.0,
    t,
    method="cubic",
)

quintic = joint_trajectory(
    0.0,
    1.0,
    t,
    method="quintic",
)
```

The three trajectories share the same endpoint positions but differ in derivative behavior.

Linear interpolation has constant velocity inside the interval. The default cubic trajectory begins and ends at zero velocity. The default quintic trajectory begins and ends at zero velocity and zero acceleration.

The public arrays can be plotted directly:

```python
import matplotlib.pyplot as plt

plt.plot(t, linear.q[:, 0], label="linear")
plt.plot(t, cubic.q[:, 0], label="cubic")
plt.plot(t, quintic.q[:, 0], label="quintic")
plt.legend()
```

The same comparison can be made with `qd` and `qdd`.

## 2. Multi-joint quintic trajectory

```python
traj = joint_trajectory(
    [0.0, 0.3, -0.5],
    [0.8, -0.2, 0.4],
    t,
    method="quintic",
)
```

The time-major result is:

```text
traj.q.shape == (101, 3)
```

Nonzero derivative conditions can also be prescribed explicitly:

```python
traj = joint_trajectory(
    [0.0, 0.3, -0.5],
    [0.8, -0.2, 0.4],
    t,
    method="quintic",
    qd0=[0.1, 0.0, 0.0],
    qdf=[0.0, 0.0, -0.1],
)
```

## 3. Cartesian trajectory plus inverse kinematics

Generate a Cartesian position trajectory:

```python
from moro.trajectory import position_trajectory

cart = position_trajectory(
    [1.5, 0.0, 0.0],
    [1.2, 0.6, 0.0],
    t,
    method="quintic",
)
```

Then pass only the position samples to the IK layer:

```python
from moro.inverse_kinematics import solve_position_trajectory

ik = solve_position_trajectory(
    robot,
    cart.p,
    q0=q_initial,
)
```

The time vector remains available as `cart.t`, while the IK result stores the solved joint configurations.

## 4. Joint trajectory plus visualization

A `JointTrajectory` can be animated directly because `traj.q` is a numerical matrix with shape `(N, robot.dof)`:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
viz.animate(traj.q)
```

The visualization layer accepts the matrix structurally and does not import or depend on `JointTrajectory`.

For nonuniform `traj.t`, the animation still uses the backend interval controls; exact variable-time playback is outside Moro 0.5.0.
