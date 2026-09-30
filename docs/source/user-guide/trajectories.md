# Trajectory Generation

Moro 0.5.0 provides numerical point-to-point trajectory generation in joint space and Cartesian position space.

The trajectory module is independent of `Robot`: it does not inspect robot geometry, joint types, joint limits, inverse kinematics, dynamics, collisions, or visualization state.

## Path versus trajectory

A path describes geometry. A trajectory adds an explicit time parametrization:

$$
x=x(t).
$$

Two trajectories may follow the same path with different timing.

## Joint-space trajectories

Use:

```python
import numpy as np
from moro.trajectory import joint_trajectory

t = np.linspace(0.0, 2.0, 101)

traj = joint_trajectory(
    [0.0, 0.2, -0.4],
    [0.8, -0.3, 0.5],
    t,
    method="quintic",
)
```

The result stores time-major arrays:

```text
traj.t.shape   == (N,)
traj.q.shape   == (N, dof)
traj.qd.shape  == (N, dof)
traj.qdd.shape == (N, dof)
```

Useful derived properties are:

```python
traj.samples
traj.duration
traj.dof
```

For one degree of freedom, scalar endpoints are accepted:

```python
traj = joint_trajectory(0.0, 1.0, t)
```

but the result still has shape `(N, 1)`.

## Cartesian position trajectories

Use:

```python
from moro.trajectory import position_trajectory

cart = position_trajectory(
    [0.0, 0.0, 0.0],
    [1.0, 0.5, 0.0],
    t,
)
```

Cartesian outputs always retain three components:

```text
cart.p.shape == (N, 3)
cart.v.shape == (N, 3)
cart.a.shape == (N, 3)
```

Two-component planar inputs are intentionally not promoted to three dimensions.

## Interpolation methods

The supported methods are:

```text
linear
cubic
quintic
```

The default is `"quintic"`.

### Linear

Linear interpolation constrains endpoint positions only:

$$
x(\tau)=x_0+(x_f-x_0)\tau.
$$

Velocity is constant and acceleration is zero inside the interval. Explicit velocity or acceleration boundary conditions are rejected.

### Cubic

Cubic interpolation constrains position and velocity at both endpoints.

If `qd0`/`qdf` or `v0`/`vf` are omitted, they default independently to zero.

Acceleration boundary conditions are not supported by the cubic method.

### Quintic

Quintic interpolation constrains position, velocity, and acceleration at both endpoints.

Omitted velocity and acceleration boundary conditions default independently to zero, making quintic interpolation a convenient smooth point-to-point default.

## Explicit time vector

The caller supplies the complete numerical time vector:

```python
t = np.linspace(5.0, 7.0, 101)
```

The initial time does not need to be zero.

Nonuniform samples are also valid:

```python
t = [0.0, 0.1, 0.4, 1.0]
```

Time values must be finite and strictly increasing.

Public velocities and accelerations are derivatives with respect to physical time, not normalized time.

## Boundary conditions

For multi-DOF joint trajectories, derivative boundary vectors must match the joint dimension exactly.

Scalar broadcasting is intentionally not supported:

```python
# Invalid for a 2-DOF trajectory
qd0 = 0.5
```

For 1-DOF trajectories, scalar derivative conditions are accepted.

## Integration with inverse kinematics

Cartesian trajectory generation and inverse kinematics remain separate operations:

```python
cart = position_trajectory(
    p0,
    pf,
    t,
    method="quintic",
)

ik = solve_position_trajectory(
    robot,
    cart.p,
    q0=q_initial,
)
```

Current position IK consumes only `cart.p`; it does not use `cart.v` or `cart.a`.

## Integration with visualization

`RobotVisualizer` accepts numerical joint vectors directly:

```python
viz.plot([0.3, -0.2])
```

and numerical joint matrices:

```python
traj = joint_trajectory(
    [0.0, 0.0],
    [0.6, -0.4],
    t,
)

viz.animate(traj.q)
```

The visualizer maps columns to `robot.qs` order.

Animation timing still uses the backend's interval-style controls. Passing `traj.q` does not imply that a nonuniform `traj.t` is reproduced exactly in time.

## Scope

Moro 0.5.0 trajectory generation intentionally does not include waypoints, splines, trapezoidal profiles, S-curves, automatic time allocation, pose interpolation, collision-aware planning, or dynamic-feasibility planning.
