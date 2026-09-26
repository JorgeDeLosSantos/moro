# Velocity inverse kinematics for a planar 2R robot

This example shows the main Moro 0.5.0 differential-kinematics workflow:

1. define a robot;
2. inspect a task Jacobian;
3. propagate a joint velocity;
4. solve a desired Cartesian velocity with the pseudoinverse;
5. compare with damped least squares.

## Define the robot

```python
from sympy import pi

from moro import Robot
from moro.abc import l1, l2, q1, q2
from moro.differential_kinematics import (
    task_jacobian,
    cartesian_velocity,
    solve_velocity_ik,
)

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

For planar position motion, select only $v_x$ and $v_y$:

```python
q = [pi / 4, -pi / 6]
parameters = {l1: 1.0, l2: 0.8}

Jxy = task_jacobian(
    robot,
    q=q,
    task=("vx", "vy"),
    parameters=parameters,
)

Jxy
```

The selected matrix is a $2\times2$ task Jacobian extracted from the full geometric Jacobian `robot.J`.

## Forward Cartesian velocity

Suppose the joint velocity is:

```python
qd = [0.4, -0.2]
```

The corresponding planar Cartesian velocity is:

```python
vxy = cartesian_velocity(
    robot,
    q=q,
    qd=qd,
    task=("vx", "vy"),
    parameters=parameters,
)

vxy
```

Moro evaluates

$$
\dot x_{task}=J_{task}(q)\dot q.
$$

## Solve a desired Cartesian velocity

Now request a planar end-effector velocity:

```python
desired = [0.10, 0.05]

solution = solve_velocity_ik(
    robot,
    q=q,
    velocity=desired,
    task=("vx", "vy"),
    parameters=parameters,
)

solution.qd
```

The returned `VelocityIKSolution` also provides diagnostics:

```python
solution.achieved_velocity
solution.residual
solution.residual_norm
solution.rank
solution.condition_number
solution.success
```

`success` means that the final residual norm is no greater than the requested `tol`.

## Damped least squares

Near a singular configuration, pseudoinverse solutions may require large joint velocities. Damped least squares provides a regularized alternative:

```python
dls = solve_velocity_ik(
    robot,
    q=q,
    velocity=desired,
    task=("vx", "vy"),
    method="dls",
    damping=0.05,
    parameters=parameters,
)

dls.qd
dls.residual_norm
```

Damping generally reduces amplification of small singular values, at the cost of some task-space tracking error.

## Apply joint-velocity limits

Velocity limits can be applied after solving:

```python
limited = solve_velocity_ik(
    robot,
    q=q,
    velocity=desired,
    task=("vx", "vy"),
    parameters=parameters,
    joint_velocity_limits=[0.5, 0.5],
)

limited.qd
limited.unconstrained_qd
limited.limited
limited.residual
```

The limits are component-wise clipping. Moro then recomputes the achieved velocity and residual from the returned `qd`; it does not redistribute saturated motion through the remaining joints.

## Key idea

```text
Robot.J
   ↓
task_jacobian()
   ↓
cartesian_velocity() / solve_velocity_ik()
```

The symbolic geometric Jacobian remains owned by `Robot`; the differential-kinematics module adds task selection, numerical inversion, and diagnostics without duplicating the underlying robot kinematics.
