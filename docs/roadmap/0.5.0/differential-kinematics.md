# Differential kinematics

## Status

Accepted as a candidate feature for Moro 0.5.0.

## Objective

Provide explicit educational tools for forward differential kinematics and velocity-level inverse kinematics:

\[
\dot{x}=J_{\mathrm{task}}(q)\dot q,
\qquad
\dot q \approx J_{\mathrm{task}}^{\dagger}(q)\dot x.
\]

The implementation is expected to live in `moro/differential_kinematics.py`.

## Task-space representation

The canonical geometric-twist ordering is

\[
(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z).
\]

Named task components are:

- `vx`, `vy`, `vz` for linear velocity;
- `wx`, `wy`, `wz` for angular velocity.

Convenience presets:

- `task="linear"` -> `("vx", "vy", "vz")`;
- `task="angular"` -> `("wx", "wy", "wz")`;
- `task="twist"` -> all six components.

Explicit subsets such as `("vx", "vy")` and `("vx", "vy", "wz")` are supported. Task dimensionality must not be inferred from velocity-vector length, and roll/pitch/yaw terminology must not be used for geometric-Jacobian angular rows.

## Preliminary public API

```python
task_jacobian(robot, q=None, *, task="twist", parameters=None)
```

Symbolic when `q=None`, numerical when a configuration is supplied.

```python
cartesian_velocity(robot, q, qd, *, task="twist", parameters=None)
```

Returns the selected task-space velocity directly.

```python
solve_velocity_ik(
    robot,
    q,
    velocity,
    *,
    task="twist",
    method="pinv",
    damping=None,
    joint_velocity_limits=None,
    parameters=None,
)
```

Supported initial methods:

- `pinv`: Moore-Penrose pseudoinverse;
- `dls`: damped least squares.

DLS is conceptually based on

\[
J_\lambda^\dagger = J^T(JJ^T+\lambda^2I)^{-1}.
\]

No separate explicit matrix-inverse method is planned.

## `VelocityIKSolution`

`solve_velocity_ik()` should return a structured result with at least:

- `qd`;
- `achieved_velocity`;
- `residual`;
- `residual_norm`;
- `rank`;
- `condition_number`;
- `method`;
- `success`;
- `message`.

Exact residual sign, success semantics, tolerances, rank policy, and handling of infinite condition numbers are deferred.

## Joint-velocity limits

Optional joint-velocity limits may initially be implemented as output saturation/clipping, provided the effect is reflected in achieved velocity and diagnostics. Constrained least-squares formulations are outside the initial scope.

## Symbolic and numerical behavior

`task_jacobian()` should preserve symbolic inspection. Velocity propagation and differential IK are primarily numerical. Symbolic robot parameters may be resolved through `parameters`, consistent with the numerical IK philosophy. General symbolic pseudoinverse solving is not a goal.

## Examples and tests

Examples should cover planar 2R differential kinematics, reduced planar tasks, redundancy, pseudoinverse versus DLS near singularities, revolute/prismatic joints, and symbolic parameters evaluated numerically.

Tests should cover task selection, dimensions, both methods, singular/near-singular behavior, limit handling, parameters, and result consistency.

## Explicitly outside 0.5.0

- null-space optimization and secondary objectives;
- obstacle or joint-limit avoidance through gradients/null-space;
- resolved-rate trajectory controllers;
- motion planning;
- acceleration-level differential kinematics;
- `Jdot`;
- dynamic control.

## Deferred detailed-design decisions

Exact validation and error messages, numerical tolerances, result invariants, saturation semantics, damping validation, rank/conditioning policy, and caching/internal helper structure remain deferred.
