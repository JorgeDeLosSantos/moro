# Trajectory generation

## Status

Accepted for Moro 0.5.0. Detailed design complete.

## Objective

Provide compact educational point-to-point trajectories in joint space and Cartesian position space, independent of robot-specific IK, limits, dynamics, and visualization.

The implementation is expected in `moro/trajectory.py`.

## Public API direction

```python
joint_trajectory(
    q0,
    qf,
    t,
    *,
    method="quintic",
    qd0=None,
    qdf=None,
    qdd0=None,
    qddf=None,
)
```

```python
position_trajectory(
    p0,
    pf,
    t,
    *,
    method="quintic",
    v0=None,
    vf=None,
    a0=None,
    af=None,
)
```

## Supported methods

- `linear`;
- `cubic`;
- `quintic` (default).

All three are intentionally retained for teaching progression.

### Linear

Endpoints only. Explicit derivative boundary conditions should be rejected rather than ignored. Documentation should explain velocity discontinuities when connecting the interval to rest.

### Cubic

Supports endpoint positions and velocities; omitted velocities default to zero. Acceleration boundary conditions are unsupported.

### Quintic

Supports endpoint positions, velocities, and accelerations; omitted velocity/acceleration conditions default to zero.

## Explicit time vector

Both functions accept an explicit strictly increasing numerical `t`. The initial API does not internally generate time grids from `duration`, `tf`, or `samples` combinations.

## Result types

`JointTrajectory` should expose at least `t`, `q`, `qd`, `qdd`, with time-major shapes:

```python
t.shape   == (N,)
q.shape   == (N, dof)
qd.shape  == (N, dof)
qdd.shape == (N, dof)
```

`PositionTrajectory` should expose `t`, `p`, `v`, `a`, with Cartesian arrays always shaped `(N, 3)`, including planar motion.

## Independence from `Robot`

Trajectory generation is mathematical and should not inspect joint types, enforce limits, solve IK, perform collision checking, or simulate dynamics.

Cartesian trajectory generation and robot-specific IK remain explicit separate operations, e.g. generate `cart.p` and then pass it to `solve_position_trajectory()`.

## Terminology

Documentation should distinguish geometric path from time-parameterized trajectory. No separate public `Path` abstraction is planned initially.

## Visualization relationship

The trajectory module should not depend on visualization, but its array conventions should allow straightforward reuse such as `visualizer.animate(traj.q)`.

## Examples and tests

Cover linear/cubic/quintic joint profiles, nonzero boundary velocities, quintic acceleration conditions, profile comparison, 3D and planar Cartesian trajectories, IK reuse, visualization reuse, endpoint conditions, shapes, time validation, method-specific boundary conditions, and known polynomial cases.

## Explicitly outside 0.5.0

- multiple waypoints;
- splines / segment blending;
- trapezoidal profiles;
- jerk-limited S-curves;
- automatic timing from limits;
- synchronized physical-limit planning;
- joint-limit enforcement;
- robot-aware validation;
- collision-aware/path planning;
- pose trajectories;
- orientation interpolation / SLERP;
- online or time-optimal generation.

## Detailed design

The implementation-level contract for this feature is finalized in:

[`docs/design/0.5.0/trajectory.md`](../../design/0.5.0/trajectory.md)

That document is the normative source for detailed API semantics, validation, numerical policy, result invariants, tests, and implementation guidance. If implementation evidence requires a contract change, update the detailed design explicitly rather than changing behavior silently.
