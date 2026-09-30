# Full-Pose Inverse Kinematics

This example introduces Moro 0.5.0 full-pose inverse kinematics through four complementary workflows.

## 1. FK to IK round trip

Define a planar 3R robot:

```python
import sympy as sp

from moro import Robot
from moro.abc import q1, q2
from moro.inverse_kinematics import solve_pose_ik

q3 = sp.symbols("q3", real=True)

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
    (0.6, 0, 0, q3, "r"),
)
```

Generate a reachable target from forward kinematics:

```python
q_target = [0.4, -0.5, 0.7]
target = robot.T.subs(dict(zip(robot.qs, q_target))).evalf()
```

Solve the full-pose problem:

```python
solution = solve_pose_ik(
    robot,
    target,
    q0=[0.2, -0.2, 0.3],
    method="lm",
    position_tol=1e-8,
    orientation_tol=1e-8,
)
```

Inspect:

```python
solution.q
solution.converged
solution.iterations
solution.position_error
solution.orientation_error
solution.achieved_pose
```

The recovered joint vector does not need to equal `q_target`; acceptance is based on the achieved pose.

## 2. Position IK versus pose IK

Position IK constrains only translation. Full-pose IK constrains translation and orientation.

For the same robot, two targets can share the same Cartesian position but impose different end-effector orientations. A position-only solve treats them as equivalent position tasks, while:

```python
solve_pose_ik(...)
```

distinguishes them through the orientation residual

$$
e_R=\operatorname{Log}(R_dR^T)^\vee.
$$

This distinction is especially important for welding, tool alignment, insertion, and other tasks where orientation matters.

## 3. Effect of position/orientation weights

Weights alter the numerical path:

```python
position_dominant = solve_pose_ik(
    robot,
    target,
    q0=[0.2, -0.2, 0.3],
    position_weight=10.0,
    orientation_weight=1.0,
)

orientation_dominant = solve_pose_ik(
    robot,
    target,
    q0=[0.2, -0.2, 0.3],
    position_weight=1.0,
    orientation_weight=10.0,
)
```

Both solves use the same physical convergence conditions:

$$
\|e_p\|\le\texttt{position_tol},
\qquad
\|e_R\|\le\texttt{orientation_tol}.
$$

Weights do not redefine success.

## 4. Restricted or underactuated target

A lower-DOF manipulator is not rejected merely because it cannot realize arbitrary six-dimensional twists.

For a single revolute joint:

```python
robot_1r = Robot((1.0, 0, 0, q1, "r"))
```

a target generated from one of its own configurations is a valid full-pose target:

```python
target_reachable = robot_1r.T.subs({q1: 0.6}).evalf()

solution = solve_pose_ik(
    robot_1r,
    target_reachable,
    q0=[0.2],
)
```

By contrast, a pose requiring translation outside the manipulator's attainable set returns a normal non-converged `PoseIKSolution` rather than raising because the target is unreachable.

## Joint limits and stagnation

Full-pose IK uses the same joint-limit format as position IK:

```python
solution = solve_pose_ik(
    robot,
    target,
    q0=[0.2, -0.2, 0.3],
    joint_limits=[
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ],
)
```

Trial configurations are clipped before forward-kinematics evaluation. If the solver repeatedly pushes against an active limit, stagnation is detected using the effective post-clipping step.

## Target format

The public target is a numerical homogeneous transform in $SE(3)$.

If orientation data starts as Euler angles, a quaternion, axis-angle, or rotation vector, convert it explicitly using `moro.transformations` before calling `solve_pose_ik()`.

Moro does not silently project invalid rotations onto $SO(3)$.
