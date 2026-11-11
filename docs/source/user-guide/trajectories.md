# Trajectory Generation

Moro provides numerical trajectory generation in joint space and Cartesian position space, including point-to-point interpolation and piecewise polynomial waypoint trajectories.

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

## Multi-waypoint trajectories

`joint_waypoint_trajectory()` and `position_waypoint_trajectory()`
generate piecewise **linear**, **cubic Hermite**, and **quintic Hermite**
trajectories. The default method is `"quintic"`, as with the point-to-point
API. Use `method="linear"` explicitly for piecewise-linear interpolation.
For cubic trajectories, `derivative_policy="shape_preserving"` estimates
interior waypoint velocities automatically while keeping the two global
endpoints at rest.

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

### Polynomial continuity and waypoint derivatives

The interpolation methods guarantee the following time-domain continuity
at waypoints when using their documented boundary conditions:

| Method | Shared boundary quantities | Continuity |
| --- | --- | --- |
| `linear` | Positions only | C0 |
| `cubic` | Position and velocity | C1 |
| `quintic` | Position, velocity and acceleration | C2 |

Linear interpolation has piecewise-constant velocity and zero acceleration
**inside** each segment. It can exhibit a velocity jump at a waypoint;
the returned acceleration is not an impulse model and cannot be used as a
physical feasibility certificate. A cubic can have discontinuous acceleration
at joins; a quintic shares the supplied acceleration at joins.

By default, `derivative_policy="zero"` sets omitted waypoint velocities and
(for quintic) accelerations to zero. This produces a **stop** at every
waypoint, not a positive-duration dwell. Supply a complete
`velocities` array (shape `(K, dof)` or `(K, 3)`) to pass
interior waypoints without stopping. With `method="quintic"`, you can
also provide an `accelerations` array of the same shape. A 1-DOF
joint trajectory accepts `(K,)` for either derivative array.

```python
from moro.trajectory import joint_waypoint_trajectory

traj = joint_waypoint_trajectory(
    [[0.0, 0.0], [0.5, 0.3], [1.2, -0.2]],
    [0.0, 1.0, 2.5],
    method="quintic",
    velocities=[[0, 0], [0.3, -0.2], [0, 0]],
    accelerations=[[0, 0], [0, 0], [0, 0]],
    samples_per_segment=51,
)
```

For `method="cubic"`, accelerations must be omitted. Linear interpolation
does not accept derivative arrays. Partial derivative matrices and scalar
broadcasting across waypoints are not supported.
Explicitly provided derivatives may cause overshoot or exceed the robot's
physical limits: no automatic time allocation or limit checking is performed.

### Shape-preserving cubic interpolation

Use `derivative_policy="shape_preserving"` with **`method="cubic"`** to
estimate waypoint velocities rather than supplying them:

```python
from moro.trajectory import joint_waypoint_trajectory

traj = joint_waypoint_trajectory(
    [0.0, 1.0, 1.8, 1.5, 2.5],
    [0.0, 1.0, 2.5, 3.0, 5.0],
    method="cubic",
    derivative_policy="shape_preserving",
    samples_per_segment=51,
)
```

The estimated interior velocities use the PCHIP-style weighted harmonic
mean of neighboring secant slopes, evaluated independently for each joint
or Cartesian coordinate. They become **zero at local reversals and
plateaus**. Unlike standard PCHIP endpoint slopes, this policy always
sets the first and final waypoint velocities to **zero**. With two
waypoints it therefore reduces to a resting-endpoint cubic.

The resulting trajectory is **C1**, and each coordinate stays between
the values at the ends of its segment (no component-wise overshoot),
including when segment durations differ. This property does *not* mean
that the complete Cartesian path is globally monotone, or that the
motion satisfies speed/acceleration limits. Acceleration need not be
continuous.

This policy rejects `velocities` and `accelerations` even if they are
explicitly all zero. It is unavailable for `linear` or `quintic`:
quintic automatic derivative estimation is not included in 0.7.0 A4.
The default remains `derivative_policy="zero"`, which may instead
accept prescribed full derivative arrays.

At each interior waypoint the stored derivative belongs to the **segment
on the right**; at the final waypoint it belongs to the last segment.
For C1/C2 continuous derivatives this distinction is immaterial, but
for discontinuous derivatives (e.g. linear velocity, cubic acceleration)
it determines the reported value. Consecutive waypoints with the same
position are allowed. Waypoint times must still be strictly increasing.

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

## Synchronized trapezoidal joint trajectories (0.7.0 T-04)

Use `joint_trapezoidal_trajectory()` to move from one joint configuration to
another with zero initial/final velocity. All joints share a normalized
progress coordinate `s(t)`, producing a straight path in **joint space**.
The generator automatically selects a triangular or trapezoidal velocity
profile that respects the prescribed per-joint speed and acceleration
magnitudes.

```python
from moro.trajectory import joint_trapezoidal_trajectory

traj = joint_trapezoidal_trajectory(
    [0.0, 0.0],
    [1.2, -0.8],
    v_max=[0.6, 0.4],
    a_max=[1.0, 0.8],
    samples=101,
)
assert traj.method == "trapezoidal"
print(traj.profile.kind, traj.duration)
# trapezoidal 2.6
```

The `v_max` and `a_max` inputs accept either one positive finite scalar
(applied to all joint coordinates) or a positive finite vector `(dof,)`.
With mixed revolute/prismatic joints, a vector is preferable because
individual limits use angular or linear units as appropriate.

By default the duration is minimal **within the symmetric rest-to-rest,
straight-joint-path profile family**, not globally time optimal for a
robot with dynamics, joint-space obstacles, or other constraints. Supply
`duration=T` to request a longer motion; Moro uniformly stretches the
minimum-duration profile, decreasing its velocities by `1/alpha` and
accelerations by `1/alpha**2`, where `alpha=T/T_min`. A shorter requested
duration is rejected.

```python
slow = joint_trapezoidal_trajectory(
    [0.0, 0.0], [1.2, -0.8],
    v_max=[0.6, 0.4], a_max=[1.0, 0.8],
    duration=5.2, t0=2.0,
)
```

The output remains a `JointTrajectory` containing time-major `t`,
`q`, `qd` and `qdd` arrays. Its optional frozen
`TrapezoidalProfileInfo` metadata includes:

- `kind`: `"triangular"`, `"trapezoidal"`, or `"stationary"`;
- `t_accel`, `t_cruise`, `t_decel`: effective phase durations;
- `minimum_duration`: the original minimum (or `None` for a hold);
- `peak_path_speed`, `path_acceleration`: effective magnitudes of the
  **dimensionless** progress coordinate's derivatives, not joint-specific
  speeds/accelerations.

A completely stationary trajectory requires an explicit positive
`duration`, for example `joint_trapezoidal_trajectory(0, 0, v_max=1,
a_max=1, duration=2)`. The profile type is then `"stationary"`.

The trajectory is C1: acceleration is piecewise constant and generally
discontinuous at phase boundaries. At interior transition instants,
sampled acceleration comes from the phase on the right; at the final
time it comes from the last phase. This method does **not** constrain jerk,
and it does not validate actuator forces, joint positions or collisions.

See the [R/RR trapezoidal teaching example](../examples/trapezoidal-trajectories.md)
for reproducible position/velocity/acceleration plots, synchronized movement,
and a same-duration comparison with quintic interpolation. For the
**corresponding inverse-dynamics effort histories**, see
[Inverse Dynamics of Joint Trajectories](../examples/inverse-dynamics-trajectories.md).
The effort analysis evaluates the ideal dynamic model, not whether
actuators can physically perform the motion.

The `profile` field is `None` for existing polynomial trajectories. It
describes the **original generation** and does not update if users edit
`traj.q` or other sampled arrays; it is not a continuous-time evaluator.
The new keyword-only dataclass field also appears in
`dataclasses.fields()` and `dataclasses.asdict()`, but the existing five
positional constructor arguments are unchanged.

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

Multi-waypoint linear, cubic and quintic interpolation is supported, including PCHIP-style interior slope estimation for cubic interpolation with resting endpoints. Joint-space point-to-point trapezoidal profiles are also supported. Automatic quintic derivative estimation, global splines, multi-waypoint trapezoidal blending, jerk-limited S-curves, general time allocation or path retiming, pose interpolation, collision-aware planning, and dynamic-feasibility planning remain outside this increment.
