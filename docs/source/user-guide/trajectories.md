# Trajectory Generation

Moro provides numerical trajectory generation in joint space and Cartesian position space, including point-to-point interpolation and piecewise-linear waypoint trajectories.

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

## Multi-waypoint trajectories (0.7.0, PR-A2)

The first multi-waypoint increment introduces `joint_waypoint_trajectory()`
and `position_waypoint_trajectory()`. **Only `method="linear"` is available
in this increment**. Cubic/quintic waypoint interpolation, prescribed derivatives,
and shape-preserving estimation are reserved for PR-A3/A4. The waypoint
function default will change to `"quintic"` when A3 lands; pass
`method="linear"` explicitly when linear interpolation is intended.

### Joint waypoints

Provide `K >= 2` waypoints as a time-major array `(K, dof)`, or a vector
`(K,)` for one degree of freedom, together with `K` strictly increasing
arrival times:

```python
from moro.trajectory import joint_waypoint_trajectory

traj = joint_waypoint_trajectory(
    [[0.0, 0.0], [0.5, 0.3], [1.2, -0.2], [0.8, 0.5]],
    [0.0, 1.0, 2.5, 4.0],
    method="linear",
    samples_per_segment=51,
)
# traj.q, traj.qd, traj.qdd have shape (151, 2).
```

The arrays are sampled results, not a persistent polynomial object. Each
waypoint time occurs once in an automatically generated sample grid.
If `S` includes both boundaries per segment, the total sample count is
`N = (K - 1) * (S - 1) + 1`. With no sampling argument,
`samples_per_segment=51` is assumed.

### Cartesian-position waypoints

`position_waypoint_trajectory()` accepts only `(K, 3)` waypoint arrays,
including planar paths with an explicit zero z-coordinate.

```python
from moro.trajectory import position_waypoint_trajectory

cart = position_waypoint_trajectory(
    [[0, 0, 0], [1, 0.5, 0], [2, 0.25, 0]],
    [0, 1, 3],
    method="linear",
    samples_per_segment=11,
)
# cart.p, cart.v, cart.a have shape (21, 3).
```

### Explicit sample times

Instead of `samples_per_segment`, provide `sample_times`:

```python
traj = joint_waypoint_trajectory(
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 3.0],
    method="linear",
    sample_times=[0.0, 0.25, 1.25, 2.75, 3.0],
)
```

The supplied times must be finite, strictly increasing, and include exactly
the start and end times of the full motion. Interior waypoint times **need
not** appear among the output samples; the underlying interpolation still
passes through them. `sample_times` and `samples_per_segment` are mutually
exclusive.

### Segment joins and limitations

Linear interpolation has continuous position (C0), piecewise-constant
velocity, and zero acceleration **inside** each segment. Velocity may jump at
an interior waypoint, so this output does not model instantaneous physical
acceleration or actuator feasibility.

At each interior waypoint the stored derivative belongs to the **segment
on the right**; at the final waypoint it belongs to the last segment.
Consecutive waypoints with the same position are allowed, including
stationary segments. Waypoint times must still be strictly increasing.
The first increment does not accept explicit velocities or accelerations;
`derivative_policy="zero"` is accepted as the default but does not force
linear-segment velocities to zero.

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

The current multi-waypoint API is limited to linear position interpolation; cubic/quintic waypoint interpolation, derivative estimation, splines, trapezoidal profiles, S-curves, automatic time allocation, pose interpolation, collision-aware planning, and dynamic-feasibility planning remain outside this first increment.
